# H100 Training Execution Instructions

Since training 300 epochs will take a long time, you should run the script in the background so it continues even if your SSH session disconnects or you close your laptop.

## 1. Prepare Directory

Make sure the log directory exists first:

```bash
mkdir -p models/train/logs
```

## 2. Start Training (Background Mode)

Run the following command:

```bash
nohup python3 train.py > models/train/logs/output.log 2>&1 &
```

### Explanation:
- `nohup`: Keeps the command running after logout.
- `> models/train/logs/output.log`: Saves all console output (print statements) to this file.
- `2>&1`: Redirects errors to the same log file.
- `&`: Runs the process in the background immediately.

---

## 3. Check Progress

To see the logs in real-time, use:

```bash
tail -f models/train/logs/output.log
```
*(Press `Ctrl+C` to stop watching without stopping training)*

---

## 4. Monitor Process

To check if the process is still running:

```bash
ps aux | grep train.py
```

To stop the training manually:
```bash
kill <PID>
```
*(Replace `<PID>` with the process ID number from the `ps` command)*
