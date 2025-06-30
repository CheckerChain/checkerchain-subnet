import os
import sys
import logging
import subprocess
import argparse
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("autoupdate.log"), logging.StreamHandler()],
)
logger = logging.getLogger("autoupdater")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-update Checkerchain repository")
    parser.add_argument(
        "--repo-path",
        type=str,
        default=os.getcwd(),
        help="Path to the repository (default: current directory)",
    )
    parser.add_argument(
        "--branch", type=str, default="main", help="Branch to pull from (default: main)"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default="requirements.txt",
        help="Path to requirements file (default: requirements.txt)",
    )
    parser.add_argument(
        "--restart-command",
        type=str,
        default=None,
        help="Command to restart the application (e.g., 'pm2 restart checkerchain-vali')",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=f"{os.getcwd()}/.venv",
        help="Path to virtual environment (optional)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if not on the default branch",
    )
    return parser.parse_args()


def run_command(command, cwd=None):
    logger.info(
        f"Running command: {command} {f"in {cwd}" if cwd else 'current directory'}"
    )
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error output: {e.stderr}")
        return None


def get_current_branch(repo_path):
    """Get the name of the current branch."""
    return run_command("git rev-parse --abbrev-ref HEAD", cwd=repo_path)


def is_on_default_branch(repo_path, default_branch):
    """Check if the repository is currently on the default branch."""
    current_branch = get_current_branch(repo_path)

    if current_branch is None:
        logger.error("Failed to determine current branch")
        return False

    is_default = current_branch == default_branch
    if not is_default:
        logger.warning(
            f"Not on default branch. Current branch: {current_branch}, Default branch: {default_branch}"
        )
    else:
        logger.info(f"On default branch: {default_branch}")

    return is_default


def check_for_updates(repo_path, branch):
    logger.info(f"Checking for updates in {repo_path} on branch {branch}")

    if run_command("git fetch origin", cwd=repo_path) is None:
        return False

    local_commit = run_command("git rev-parse HEAD", cwd=repo_path)
    remote_commit = run_command(f"git rev-parse origin/{branch}", cwd=repo_path)

    if local_commit != remote_commit:
        logger.info(f"Updates available: {local_commit} → {remote_commit}")
        return True
    else:
        logger.info("No updates available")
        return False


def update_repository(repo_path, branch):
    """Pull the latest changes from the remote repository."""
    logger.info(f"Updating repository from branch {branch}")
    result = run_command(f"git pull origin {branch}", cwd=repo_path)
    if result is not None:
        logger.info("Repository updated successfully")
        return True
    return False


def update_dependencies(repo_path, requirements_file, venv=None):
    """Update Python dependencies."""
    requirements_path = os.path.join(repo_path, requirements_file)
    if not os.path.exists(requirements_path):
        logger.warning(f"Requirements file not found: {requirements_path}")
        return False

    logger.info(f"Updating dependencies from {requirements_file}")

    pip_cmd = "pip"
    if venv:
        if sys.platform == "win32":
            pip_cmd = os.path.join(venv, "Scripts", "pip")
        else:
            pip_cmd = os.path.join(venv, "bin", "pip")

    result = run_command(f"{pip_cmd} install -e .", cwd=repo_path)
    if result is not None:
        logger.info("Dependencies updated successfully")
        return True
    return False


def restart_application(restart_command, repo_path):
    """Restart the application using the provided command."""
    if not restart_command:
        return True

    logger.info(f"Restarting application with command: {restart_command}")
    result = run_command(restart_command, cwd=repo_path)
    if result is not None:
        logger.info("Application restarted successfully")
        return True
    return False


def run_migrations(repo_path):
    # run alembic migrations
    logger.info("Running database migrations")
    alembic_cmd = "alembic upgrade head"
    if run_command(alembic_cmd, cwd=repo_path) is not None:
        logger.info("Database migrations completed successfully")
        return True
    logger.error("Failed to run database migrations")
    return False


def main():
    args = parse_arguments()

    logger.info(
        f"=== Auto-update started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
    )

    # Change to repository directory
    if not os.path.exists(args.repo_path):
        logger.error(f"Repository path does not exist: {args.repo_path}")
        return 1

    # Check if it's a git repository
    if not os.path.exists(os.path.join(args.repo_path, ".git")):
        logger.error(f"Not a git repository: {args.repo_path}")
        return 1

    if not args.force and not is_on_default_branch(args.repo_path, args.branch):
        logger.error(
            f"Not on the default branch ({args.branch}). Use --force to update anyway."
        )
        return 1

    # Check for updates
    if check_for_updates(args.repo_path, args.branch):
        # Update repository
        if not update_repository(args.repo_path, args.branch):
            logger.error("Failed to update repository")
            return 1

        # Update dependencies
        if not update_dependencies(args.repo_path, args.requirements, args.venv):
            logger.error("Failed to update dependencies")
            return 1

        if not run_migrations(args.repo_path):
            logger.error("Failed to run database migrations")
            return 1

        # Restart application if needed
        if not restart_application(args.restart_command, args.repo_path):
            logger.error("Failed to restart application")
            return 1

    logger.info(
        f"=== Auto-update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
