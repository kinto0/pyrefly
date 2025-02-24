/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::backtrace::Backtrace;
use std::env::args_os;
use std::process::ExitCode;

use clap::Parser;
use pyre2::get_args_expanded;
use pyre2::init_tracing;
use pyre2::run::run_command;
use pyre2::run::Command;
use pyre2::run::CommandExitStatus;

#[derive(Debug, Parser)]
#[command(name = "pyre2")]
#[command(about = "Next generation of Pyre type checker", long_about = None)]
struct Args {
    /// Enable verbose logging.
    #[clap(long = "verbose", short = 'v', global = true)]
    verbose: bool,

    /// Set this to true to run profiling of fast jobs.
    /// Will run the command repeatedly.
    #[clap(long = "profiling", global = true, hide = true)]
    profiling: bool,

    #[command(subcommand)]
    command: Command,
}

fn exit_on_panic() {
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("Thread panicked, shutting down: {}", info);
        eprintln!("Backtrace:\n{}", Backtrace::force_capture());
        std::process::exit(1);
    }));
}

fn to_exit_code(status: CommandExitStatus) -> ExitCode {
    match status {
        CommandExitStatus::Success => ExitCode::SUCCESS,
        CommandExitStatus::UserError => ExitCode::FAILURE,
    }
}

/// Run based on the command line arguments.
fn run() -> anyhow::Result<ExitCode> {
    let args = Args::parse_from(get_args_expanded(args_os())?);
    if args.profiling {
        loop {
            let _ = run_command(args.command.clone(), false);
        }
    } else {
        init_tracing(args.verbose, false);
        run_command(args.command, true).map(to_exit_code)
    }
}

pub fn main() -> ExitCode {
    exit_on_panic();
    let res = run();
    match res {
        Ok(code) => code,
        Err(e) => {
            // If you return a Result from main, and RUST_BACKTRACE=1 is set, then
            // it will print a backtrace - which is not what we want.
            eprintln!("{:#}", e);
            ExitCode::FAILURE
        }
    }
}
