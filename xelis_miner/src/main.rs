pub mod config;

use std::{
    time::Duration,
    sync::atomic::{
        AtomicU64,
        Ordering,
        AtomicUsize,
        AtomicBool
    },
    thread
};
use crate::config::DEFAULT_DAEMON_ADDRESS;
use fern::colors::Color;
use futures_util::{StreamExt, SinkExt};
use serde::{Serialize, Deserialize};
use tokio::{
    sync::{
        broadcast,
        mpsc,
        Mutex
    },
    select,
    time::Instant,
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{
        Message,
        Error as TungsteniteError
    }
};
use xelis_common::{
    api::daemon::{
        GetMinerWorkResult,
        SubmitMinerWorkParams,
    },
    async_handler,
    block::MinerWork,
    config::VERSION,
    crypto::{
        Address,
        Hash,
        PublicKey,
    },
    difficulty::{
        compute_difficulty_target,
        check_difficulty_against_target,
        difficulty_from_hash,
        Difficulty
    },
    prompt::{
        self,
        command::CommandManager,
        LogLevel,
        Prompt,
        ShareablePrompt
    },
    serializer::Serializer,
    time::get_current_time_in_millis,
    utils::{
        format_difficulty,
        format_hashrate,
        sanitize_daemon_address,
        spawn_task
    }
};
use clap::Parser;
use log::{
    debug,
    info,
    warn,
    error,
};
use anyhow::{
    Result,
    Error,
    Context,
    anyhow
};
use lazy_static::lazy_static;
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand::rngs::ThreadRng;
use primitive_types::U256;

extern crate ta;
use ta::{
    indicators::SimpleMovingAverage,
    Next,
};
use obfstr::obfstr;

#[derive(Parser)]
#[clap(version = VERSION, about = "XELIS: An innovate cryptocurrency with BlockDAG and Homomorphic Encryption enabling Smart Contracts")]
#[command(styles = xelis_common::get_cli_styles())]
pub struct MinerConfig {
    /// Wallet address to mine and receive block rewards on
    #[clap(short, long)]
    miner_address: Option<Address>,
    /// Daemon address to connect to for mining
    #[clap(long, default_value_t = String::from(DEFAULT_DAEMON_ADDRESS))]
    daemon_address: String,
    /// Set log level
    #[clap(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,
    /// Enable the benchmark mode
    #[clap(long)]
    benchmark: bool,
    /// Iterations to run the benchmark
    #[clap(long, default_value_t = 10)]
    iterations: usize,
    /// Num GPUs to use. Uses first N. -1 means use all GPUs.
    #[clap(long, default_value_t = 32)]
    gpu_count: u16,
    /// Benchmark all parameter configurations, in power of 2
    #[clap(long, default_value_t = false)]
    bench_all: bool,
    /// Disable the log file
    #[clap(long)]
    disable_file_logging: bool,
    /// Log filename
    /// 
    /// By default filename is xelis-miner.log.
    /// File will be stored in logs directory, this is only the filename, not the full path.
    /// Log file is rotated every day and has the format YYYY-MM-DD.xelis-miner.log.
    #[clap(default_value_t = String::from("xelis-miner.log"))]
    filename_log: String,
    /// Logs directory
    /// 
    /// By default it will be logs/ of the current directory.
    /// It must end with a / to be a valid folder.
    #[clap(long, default_value_t = String::from("logs/"))]
    logs_path: String,
    /// Number of Threads per GPU
    #[clap(short, long, default_value_t = 2)]
    num_threads_per_gpu: u16,
    /// Worker name to be displayed on daemon side
    #[clap(short, long, default_value_t = String::from("default"))]
    worker: String,
    /// Batch size for GPU work
    #[clap(short, long, default_value_t = 16384)]
    batch_size: usize
}

#[derive(Clone)]
enum ThreadNotification<'a> {
    NewJob(MinerWork<'a>, Difficulty, u64), // block work, difficulty, height
    WebSocketClosed, // WebSocket connection has been closed
    Exit // all threads must stop
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")] 
pub enum SocketMessage {
    NewJob(GetMinerWorkResult),
    BlockAccepted,
    BlockRejected(String)
}

const CLI_UPDATE_INTERVAL_US: u64 = 1000;
const SERVER_POLLING_INTERVAL_MS: u64 = 10;
const EMA_SMOOTHING_FACTOR: usize = 20;
const DEV_FEE_PERCENT: u64 = 1;

static WEBSOCKET_CONNECTED: AtomicBool = AtomicBool::new(false);
static CURRENT_TOPO_HEIGHT: AtomicU64 = AtomicU64::new(0);
static BLOCKS_FOUND: AtomicUsize = AtomicUsize::new(0);
static BLOCKS_REJECTED: AtomicUsize = AtomicUsize::new(0);
static HASHRATE_COUNTER: AtomicUsize = AtomicUsize::new(0);

lazy_static! {
    static ref HASHRATE_LAST_TIME: Mutex<Instant> = Mutex::new(Instant::now());
    static ref HASHRATE_SMA: Mutex<SimpleMovingAverage> = Mutex::new(SimpleMovingAverage::new(EMA_SMOOTHING_FACTOR).unwrap());
    static ref DEV_RNG: Mutex<StdRng> = Mutex::new(StdRng::from_entropy());
}


extern "C" {
    //fn xelis_hash_cuda(input: *const u8, count: usize,
    //    output: *mut u8, state: i32) -> i32;
    fn initialize_cuda(count: usize, num_states: i32) -> i32;
    fn deinitialize_cuda() -> i32;
    fn xelis_hash_cuda_nonce(base_header: *const u8, nonce_start: *mut u64,
        batch_size: usize, output_hash: *mut u8,
        output_nonce: *mut u64,
        difficulty: *const u8, gpu_id: i32, state: i32) -> i32;
}


#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let config: MinerConfig = MinerConfig::parse();
    let prompt = Prompt::new(config.log_level, &config.logs_path, &config.filename_log, config.disable_file_logging)?;

    let thread_per_gpu = config.num_threads_per_gpu;
    let batch_size = config.batch_size;

    if config.benchmark {
        info!("Benchmark mode enabled, miner will try up to {} gpus, {} threads each, {} batch size",
            config.gpu_count, thread_per_gpu, batch_size);
        benchmark(config.iterations, config.bench_all as bool,
            batch_size, config.gpu_count as usize, thread_per_gpu as usize);
        info!("Benchmark finished");
        return Ok(())
    }

    let threads: usize;
    let mut num_gpus: u16;
    unsafe {
        num_gpus = initialize_cuda(batch_size, thread_per_gpu as i32) as u16;
    }
    if num_gpus <= 0 {
        error!("Failed to initialize CUDA: {}", num_gpus);
        return Err(anyhow!("Failed to initialize CUDA: {}", num_gpus));
    }
    num_gpus = std::cmp::min(num_gpus, config.gpu_count);
    threads = (num_gpus * thread_per_gpu) as usize;

    let address = config.miner_address.ok_or_else(|| Error::msg("No miner address specified"))?;
    info!("Miner address: {}", address);

    // broadcast channel to send new jobs / exit command to all threads
    let (sender, _) = broadcast::channel::<ThreadNotification>(threads as usize);
    // mpsc channel to send from threads to the "communication" task.
    let (block_sender, block_receiver) = mpsc::channel::<MinerWork>(threads as usize);
    let mut id: u16 = 0;
    for gpu_id in 0..num_gpus {
        for state_id in 0..thread_per_gpu {
            debug!("Starting thread #{}", id);
            if let Err(e) = start_thread(id, gpu_id, state_id, batch_size, sender.subscribe(), block_sender.clone()) {
                error!("Error while creating Mining Thread #{}: {}", id, e);
            }
            id += 1;
        }
    }

    //let dev_fee_wallet_str: String = obfstr!("xet:5g2sddyfnc6u8xddhpah0kl6qch2rujluytj5kw9xjudrrs6jgzqq0tk8fx").to_string();
    let dev_fee_wallet_str: String = obfstr!("xel:jnwns8nffawr23vszhh27hakrxzuuj2vhljfltsc8u08qrnvg5xsqek2wr6").to_string();

    let dev_address: Address = match Address::from_string(&dev_fee_wallet_str) {
        Ok(_dev_address) => _dev_address,
        Err(e) => {
            debug!("Invalid fee address: {}", e);
        return Err(anyhow!("Invalid fee address: {}", e));
        }
    };
    let dev_key = dev_address.to_public_key();

    // start communication task
    let task = spawn_task("communication", communication_task(config.daemon_address, sender.clone(),
        block_receiver, address, config.worker, dev_key));

    if let Err(e) = run_prompt(prompt).await {
        error!("Error on running prompt: {}", e);
    }

    // send exit command to all threads to stop
    if let Err(_) = sender.send(ThreadNotification::Exit) {
        debug!("Error while sending exit message to threads");
    }

    // stop the communication task
    task.abort();

    /*unsafe {
        let _ = deinitialize_cuda();
    }*/

    Ok(())
}

fn benchmark(iterations: usize, bench_all: bool, max_batch_size: usize, max_gpus: usize, max_states: usize) {

    let mut num_gpus: usize;
    info!("{0: <5} | {1: <9} | {2: <8} | {3: <12} | {4: <16} | {5: <13} | {6: <13}", "GPUS", "Batch Size", "Threads", "Total Time", "Total Iterations", "Time/PoW (ms)", "Hashrate");
    // Loop over powers of 2 for num_states
    let mut num_states = (if bench_all {1} else {max_states}) as usize;
    while num_states <= max_states {
        // Loop over powers of 2 for batch_size
        let mut batch_size = if bench_all {1} else {max_batch_size} as usize;
        while batch_size <= max_batch_size {
            let ret: i32;
            unsafe {
                ret = initialize_cuda(batch_size, num_states as i32);
            }
            if ret <= 0 {
                batch_size *= 2;
                continue;
            }
            num_gpus = std::cmp::min(max_gpus, ret as usize);

            let start = Instant::now();
            let mut handles = vec![];
            for gpu_id in 0..num_gpus {
                for state_id in 0..num_states {
                    let mut job = MinerWork::new(Hash::zero(), get_current_time_in_millis());
                    let handle = thread::spawn(move || {
                        for _ in 0..iterations {
                            let diff = U256::max_value();
                            let mut base_nonce = 0;
                            get_pow_hash(&job, &mut base_nonce,
                                batch_size,
                                gpu_id as u16, state_id as u16, &diff);

                            job.set_timestamp(get_current_time_in_millis()).unwrap();
                        }
                    });
                    handles.push(handle);
                }
            }

            for handle in handles { // wait on all threads
                handle.join().unwrap();
            }
            let duration = start.elapsed().as_millis();
            if duration > 200 {
                let hashrate = format_hashrate(1000f64 / (duration as f64 / (num_gpus*num_states*iterations*batch_size) as f64));
                info!("{0: <5} | {1: <9} | {2: <8} | {3: <12} | {4: <16} | {5: <13} | {6: <13}",
                    num_gpus, batch_size, num_states,
                    duration, num_gpus*num_states*iterations*batch_size,
                    duration/(num_gpus*num_states*iterations*batch_size) as u128, hashrate);
            }

            unsafe {
                let _ = deinitialize_cuda();
            }
            batch_size *= 2;
        }
        num_states *= 2;
    }
}

// this Tokio task will runs indefinitely until the user stop himself the miner.
// It maintains a WebSocket connection with the daemon and notify all threads when it receive a new job.
// Its also the task who have the job to send directly the new block found by one of the threads.
// This allow mining threads to only focus on mining and receiving jobs through memory channels.
async fn communication_task(daemon_address: String, job_sender: broadcast::Sender<ThreadNotification<'_>>,
        mut block_receiver: mpsc::Receiver<MinerWork<'_>>, address: Address, worker: String,
        dev_key: PublicKey) {
    info!("Starting communication task");

    let daemon_address = sanitize_daemon_address(&daemon_address);
    'main: loop {
        info!("Trying to connect to {}", daemon_address);
        let client = match connect_async(format!("{}/getwork/{}/{}", daemon_address, address.to_string(), worker)).await {
            Ok((client, response)) => {
                let status = response.status();
                if status.is_server_error() || status.is_client_error() {
                    error!("Error while connecting to {}, got an unexpected response: {}", daemon_address, status.as_str());
                    warn!("Trying to connect to WebSocket again in 10 seconds...");
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    continue 'main;
                }
                client
            },
            Err(e) => {
                if let TungsteniteError::Http(e) = e {
                    let body: String = e.into_body()
                        .map_or(
                            "Unknown error".to_owned(),
                            |v| String::from_utf8_lossy(&v).to_string()
                        );
                    error!("Error while connecting to {}, got an unexpected response: {}", daemon_address, body);
                } else {
                    error!("Error while connecting to {}: {}", daemon_address, e);
                }

                warn!("Trying to connect to WebSocket again in 10 seconds...");
                tokio::time::sleep(Duration::from_secs(10)).await;
                continue 'main;
            }
        };
        WEBSOCKET_CONNECTED.store(true, Ordering::SeqCst);
        info!("Connected successfully to {}", daemon_address);
        let (mut write, mut read) = client.split();
        loop {
            select! {
                Some(message) = read.next() => { // read all messages from daemon
                    debug!("Received message from daemon: {:?}", message);
                    match handle_websocket_message(message, &job_sender, dev_key.clone()).await {
                        Ok(exit) => {
                            if exit {
                                debug!("Exiting communication task");
                                break;
                            }
                        },
                        Err(e) => {
                            error!("Error while handling message from WebSocket: {}", e);
                            break;
                        }
                    }
                },
                Some(work) = block_receiver.recv() => { // send all valid blocks found to the daemon
                    info!("submitting new block found...");
                    let submit = serde_json::json!(SubmitMinerWorkParams { miner_work: work.to_hex() }).to_string();
                    if let Err(e) = write.send(Message::Text(submit)).await {
                        error!("Error while sending the block found to the daemon: {}", e);
                        break;
                    }
                    debug!("Block found has been sent to daemon");
                }
            }
        }

        WEBSOCKET_CONNECTED.store(false, Ordering::SeqCst);
        if job_sender.send(ThreadNotification::WebSocketClosed).is_err() {
            error!("Error while sending WebSocketClosed message to threads");
        }

        warn!("Trying to connect to WebSocket again in 10 seconds...");
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}

async fn handle_websocket_message(message: Result<Message, TungsteniteError>,
        job_sender: &broadcast::Sender<ThreadNotification<'_>>,
        dev_key: PublicKey) -> Result<bool, Error> {

    let mut dev_rng = DEV_RNG.lock().await;
    match message? {
        Message::Text(text) => {
            debug!("new message from daemon: {}", text);
            match serde_json::from_slice::<SocketMessage>(text.as_bytes())? {
                SocketMessage::NewJob(job) => {
                    info!("New job received: difficulty {} at height {}", format_difficulty(job.difficulty), job.height);
                    let mut block = MinerWork::from_hex(job.template).context("Error while decoding new job received from daemon")?;
                    CURRENT_TOPO_HEIGHT.store(job.topoheight, Ordering::SeqCst);
                    let dev_uniform: Uniform<u64> = Uniform::new(0, 100);

                    if dev_rng.sample(dev_uniform) < DEV_FEE_PERCENT{
                        //change miner pubkey
                        block.set_miner(std::borrow::Cow::Owned(dev_key));
                        debug!("Set work pubkey to dev fee");
                    }

                    if let Err(e) = job_sender.send(ThreadNotification::NewJob(block, job.difficulty, job.height)) {
                        error!("Error while sending new job to threads: {}", e);
                    }
                },
                SocketMessage::BlockAccepted => {
                    BLOCKS_FOUND.fetch_add(1, Ordering::SeqCst);
                    info!("Block submitted has been accepted by network !");
                },
                SocketMessage::BlockRejected(err) => {
                    BLOCKS_REJECTED.fetch_add(1, Ordering::SeqCst);
                    error!("Block submitted has been rejected by network: {}", err);
                }
            }
        },
        Message::Close(reason) => {
            let reason: String = if let Some(reason) = reason {
                reason.to_string()
            } else {
                "No reason".into()
            };
            warn!("Daemon has closed the WebSocket connection with us: {}", reason);
            return Ok(true);
        },
        _ => {
            warn!("Unexpected message from WebSocket");
            return Ok(true);
        }
    };

    Ok(false)
}

///////////////////

const HASH_SIZE: usize = 32;

fn get_pow_hash(job: &MinerWork, base_nonce: &mut u64, count: usize,
        gpu_id: u16, state_id: u16, difficulty: &U256) -> (xelis_common::crypto::Hash, u64){
    debug!("Starting POW: {}", count);
    let mut base_header = job.to_bytes().to_vec();
    base_header.resize(200, 0);

    let mut hash_bytes = [0u8; HASH_SIZE];
    let mut nonce: u64 = 0;

    let mut diff_bytes = [0u8; 32];
    difficulty.to_big_endian(&mut diff_bytes);

    let result: i32;
    unsafe {
        result = xelis_hash_cuda_nonce(base_header.as_ptr() as *const u8,
            base_nonce as *mut u64,
            count,
            hash_bytes.as_mut_ptr() as *mut u8,
            &mut nonce as *mut u64,
            diff_bytes.as_mut_ptr() as *const u8,
            gpu_id as i32,
            state_id as i32);
    };
    debug!("Completed GPU call {}", result);
    let hash = xelis_common::crypto::Hash::from_bytes(&hash_bytes).unwrap();

    (hash, nonce)
}

fn start_thread(id: u16, gpu_id: u16, state_id: u16, batch_size: usize, mut job_receiver: broadcast::Receiver<ThreadNotification<'static>>, block_sender: mpsc::Sender<MinerWork<'static>>) -> Result<(), Error> {
    let builder = thread::Builder::new().name(format!("Mining Thread #{} gpu {}", id, gpu_id));
    builder.spawn(move || {
        //let mut ref_job: MinerWork;
        let mut job: MinerWork;
        let mut hash: Hash;
        let mut nonce: u64;
        let uniform: Uniform<u64> = Uniform::new(0, u64::MAX);
        let mut rng: ThreadRng = rand::thread_rng();


        info!("Mining Thread #{}: started", id);
        'main: loop {
            let message = match job_receiver.blocking_recv() {
                Ok(message) => message,
                Err(e) => {
                    error!("Error on thread #{} while waiting on new job: {}", id, e);
                    // Channel is maybe lagging, try to empty it
                    while job_receiver.len() > 1 {
                        let _ = job_receiver.blocking_recv();
                    }
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            match message {
                ThreadNotification::WebSocketClosed => {
                    // wait until we receive a new job, check every 100ms
                    while job_receiver.is_empty() {
                        thread::sleep(Duration::from_millis(SERVER_POLLING_INTERVAL_MS));
                    }
                }
                ThreadNotification::Exit => {
                    info!("Exiting Mining Thread #{}...", id);
                    break 'main;
                },
                ThreadNotification::NewJob(new_job, expected_difficulty, height) => {
                    debug!("Mining Thread #{} received a new job", id);
                    job = new_job;
                    // set thread id in extra nonce for more work spread between threads
                    // u16 support up to 65535 threads
                    job.set_thread_id_u16(id);


                    let difficulty_target = match compute_difficulty_target(&expected_difficulty) {
                        Ok(value) => value,
                        Err(e) => {
                            error!("Mining Thread #{}: error on difficulty target computation: {}", id, e);
                            continue 'main;
                        }
                    };
                    debug!("Original job {}", job.to_hex());

                    let mut base_nonce = job.nonce() + uniform.sample(&mut rng);

                    (hash, nonce) = get_pow_hash(&job, &mut base_nonce,
                        batch_size, gpu_id, state_id, &difficulty_target);
                    while !check_difficulty_against_target(&hash, &difficulty_target) {
                        job.set_timestamp(get_current_time_in_millis()).unwrap();
                        HASHRATE_COUNTER.fetch_add(batch_size as usize, Ordering::SeqCst);

                        if !job_receiver.is_empty() {
                            continue 'main;
                        }

                        (hash, nonce) = get_pow_hash(&job, &mut base_nonce,
                            batch_size, gpu_id, state_id, &difficulty_target);
                    }

                    info!("Found nonce {}", nonce);
                    let nonce_ret = job.set_nonce(nonce.to_be());
                    debug!("With new nonce {}, ret {:?}", job.to_hex(), nonce_ret);

                    // compute the reference hash for easier finding of the block
                    info!("Thread #{}: block {} found at height {} with difficulty {}", id, hash, height, format_difficulty(difficulty_from_hash(&hash)));
                    if let Err(_) = block_sender.blocking_send(job) {
                        error!("Mining Thread #{}: error while sending block found with hash {}", id, hash);
                        continue 'main;
                    }
                    debug!("Job sent to communication task");
                }
            };
        }
        info!("Mining Thread #{}: stopped", id);
    })?;
    Ok(())
}

async fn run_prompt(prompt: ShareablePrompt) -> Result<()> {
    let command_manager = CommandManager::new(prompt.clone());
    command_manager.register_default_commands()?;

    let closure = |_: &_, _: _| async {
        let topoheight_str = format!(
            "{}: {}",
            prompt::colorize_str(Color::Yellow, "TopoHeight"),
            prompt::colorize_string(Color::Green, &format!("{}", CURRENT_TOPO_HEIGHT.load(Ordering::SeqCst))),
        );
        let blocks_found = format!(
            "{}: {}",
            prompt::colorize_str(Color::Yellow, "Accepted"),
            prompt::colorize_string(Color::Green, &format!("{}", BLOCKS_FOUND.load(Ordering::SeqCst))),
        );
        let blocks_rejected = format!(
            "{}: {}",
            prompt::colorize_str(Color::Yellow, "Rejected"),
            prompt::colorize_string(Color::Green, &format!("{}", BLOCKS_REJECTED.load(Ordering::SeqCst))),
        );
        let status = if WEBSOCKET_CONNECTED.load(Ordering::SeqCst) {
            prompt::colorize_str(Color::Green, "Online")
        } else {
            prompt::colorize_str(Color::Red, "Offline")
        };
        let hashrate = {
            let mut last_time = HASHRATE_LAST_TIME.lock().await;
            let counter = HASHRATE_COUNTER.swap(0, Ordering::SeqCst);
            let mut sma = HASHRATE_SMA.lock().await;


            let hashrate = 1000f64 / (last_time.elapsed().as_millis() as f64 / counter as f64);
            *last_time = Instant::now();
            
            let mut sma_hr: f64 = 0f64;
            if !hashrate.is_nan() {
                sma_hr = sma.next(hashrate);
            }

            prompt::colorize_string(Color::Green, &format!("{} ({})",
                format_hashrate(sma_hr), format_hashrate(hashrate)))
        };

        Ok(
            format!(
                "{} | {} | {} | {} | {} | {} {} ",
                prompt::colorize_str(Color::Blue, "XELIS Miner"),
                topoheight_str,
                blocks_found,
                blocks_rejected,
                hashrate,
                status,
                prompt::colorize_str(Color::BrightBlack, ">>")
            )
        )
    };

    prompt.start(Duration::from_millis(CLI_UPDATE_INTERVAL_US), Box::new(async_handler!(closure)), Some(&command_manager)).await?;
    Ok(())
}