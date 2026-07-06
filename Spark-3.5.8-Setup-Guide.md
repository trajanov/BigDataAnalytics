# Setting Up Apache Spark 3.5.8 on Your Laptop (Beginner Guide)

This guide walks you through installing **PySpark 3.5.8** and running it inside **Visual Studio Code (VS Code)** notebooks. We use the **simple installation method**: we install Spark through Python's `pip` (the `pyspark` package), so you do **not** need to download and configure Spark by hand.

By the end you will be able to open any `.ipynb` notebook in this course, pick the **"Python (spark) 3.5.8"** kernel, and run Spark code.

> Follow the steps **in order**. Don't skip any. Copy-paste the commands exactly. It should take about 20–30 minutes.

---

## What we are going to install

| Tool | Why we need it |
|------|----------------|
| **Miniconda** | Manages Python and creates isolated "environments" so this course doesn't break your other Python projects. |
| **Python 3.11** | The Python version PySpark 3.5.8 works best with. (Newer Python like 3.13/3.14 is **not** compatible.) |
| **Java 11 (JDK)** | Spark runs on the Java Virtual Machine. Spark 3.5.8 needs Java 8 or 11. **Java 17 or newer will crash it.** |
| **pyspark 3.5.8** | Spark itself (Python + the Spark engine bundled together). |
| **ipykernel** | Lets Jupyter/VS Code use this environment as a notebook "kernel". |
| **VS Code** + extensions | The editor you'll write and run notebooks in. |

> **Key idea:** We put Python, Java, and Spark all inside **one conda environment called `spark`**. Everything stays in one place, and conda configures Java for you automatically.

---

## Part 0 — Before you start

- **Disk space:** ~4 GB free.
- **Internet:** required (downloads are ~1 GB total).
- **Admin rights:** helpful but usually not required for a "just me" install.

---

## Part 1 — Install Miniconda

Miniconda is a small installer that gives you the `conda` command.

### Windows

1. Go to <https://www.anaconda.com/download/success> and under **Miniconda Installers** download the **Windows 64-bit** `.exe`.
2. Run the installer. When asked, choose **"Install for: Just Me"**.
3. Keep clicking **Next** and accept the defaults. **Do not** change the install location.
4. Finish the installation.

### macOS

1. From the same page, download the **macOS** installer that matches your chip:
   - Apple Silicon (M1/M2/M3/M4) → the **Apple Silicon (arm64)** `.pkg`
   - Older Intel Mac → the **Intel (x86_64)** `.pkg`
   - *(Not sure? Click the Apple menu → About This Mac → look at "Chip" or "Processor".)*
2. Open the `.pkg` and follow the prompts with the defaults.

### Check it worked

Open a **fresh** terminal:

- **Windows:** press the Start button, type **`Anaconda Prompt`**, and open it. *(Use Anaconda Prompt for all conda commands on Windows — not the regular Command Prompt.)*
- **macOS:** open the **Terminal** app.

Type:

```bash
conda --version
```

You should see something like `conda 24.x.x`. If you get "command not found", **close the terminal, open a new one**, and try again.

---

## Part 2 — Create the Spark environment

This single command creates an environment named `spark` with the correct Python **and** the correct Java. Copy-paste it into your terminal (Anaconda Prompt on Windows) and press Enter:

```bash
conda create -n spark -c conda-forge python=3.11 openjdk=11 ipykernel -y
```

- `-n spark` → the environment's name is **spark**
- `python=3.11` → the compatible Python
- `openjdk=11` → Java 11 (conda sets `JAVA_HOME` for you — no manual Java install!)
- `ipykernel` → so VS Code can use it as a notebook kernel

Wait for it to finish (a few minutes). Then **activate** the environment:

```bash
conda activate spark
```

Your prompt should now start with `(spark)`. **Everything from here on must be done with `(spark)` active.**

Now install **PySpark 3.5.8** with pip:

```bash
pip install pyspark==3.5.8
```

> The `==3.5.8` is important — it locks the exact version this course uses. This downloads ~300 MB and may take a minute or two.

---

## Part 3 — Windows only: install `winutils` (Hadoop helper)

> **macOS / Linux users: skip this part entirely — go to Part 4.**

On Windows, Spark needs a small helper file called `winutils.exe` (plus `hadoop.dll`) or it will throw errors when reading/writing files. This is a well-known Windows quirk, not something you did wrong.

1. Create a folder named `C:\hadoop\bin`.
2. Download two files for **Hadoop 3.3** from this trusted community repository:
   - Go to <https://github.com/cdarlint/winutils>
   - Open the folder **`hadoop-3.3.5/bin`** (or the closest `hadoop-3.3.x`).
   - Download **`winutils.exe`** and **`hadoop.dll`**.
3. Put **both** files inside `C:\hadoop\bin`.
4. Set two environment variables so Spark can find them:

   **Easiest way (permanent):**
   1. Press Start, type **"Edit the system environment variables"**, open it.
   2. Click **Environment Variables…**
   3. Under **User variables**, click **New…** and add:
      - Variable name: `HADOOP_HOME`
      - Variable value: `C:\hadoop`
   4. Still under User variables, select **`Path`** → **Edit…** → **New** → add:
      - `%HADOOP_HOME%\bin`
   5. Click **OK** on every window.

> ⚠️ **Do NOT set a `SPARK_HOME` environment variable.** Because we installed Spark via `pip`, PySpark already knows where it lives. A stray `SPARK_HOME` pointing at a different Spark version is the #1 cause of the confusing error `Constructor org.apache.spark.sql.SparkSession(...) does not exist`. If you ever added one, delete it here.

After changing environment variables, **you must fully close and reopen** any terminals and VS Code (see Part 6 note).

---

## Part 4 — Register the notebook kernel

This makes your `spark` environment show up in VS Code's kernel list.

With `(spark)` still active, run:

```bash
python -m ipykernel install --user --name spark --display-name "Python (spark) 3.5.8"
```

You should see: `Installed kernelspec spark in ...`

Verify it registered:

```bash
jupyter kernelspec list
```

You should see `spark` in the list. ✅

---

## Part 5 — Install VS Code and its extensions

1. Download and install **Visual Studio Code**: <https://code.visualstudio.com/>
2. Open VS Code.
3. Click the **Extensions** icon on the left sidebar (four squares), and install these two, both published by **Microsoft**:
   - **Python**
   - **Jupyter**
4. *(Windows only, recommended)* Also install the **Jupyter** extension's dependencies if prompted.

---

## Part 6 — Open a notebook and select the kernel

> **Important — restart first:** If you changed environment variables in Part 3, **completely quit VS Code** (close *all* windows) and reopen it. A simple "Reload Window" is **not** enough — VS Code only reads environment variables when it first starts.

1. In VS Code, open the course folder (**File → Open Folder…**) and open any `.ipynb` notebook, for example `Notebooks/Spark-Example-01-Word-Count-Example.ipynb`.
2. In the **top-right** corner of the notebook, click **Select Kernel**.
3. Choose **Jupyter Kernel** → **Python (spark) 3.5.8**.
   - *If you don't see it:* click the refresh icon, or use **Ctrl+Shift+P → "Developer: Reload Window"**, then try again.

---

## Part 7 — Test that Spark works

Create a new notebook (**File → New File → Jupyter Notebook**) or use an empty cell, select the **Python (spark) 3.5.8** kernel, and run this in a cell:

```python
import sys
import pyspark
from pyspark.sql import SparkSession

print("Python  :", sys.executable)
print("PySpark :", pyspark.__version__)

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
print("Spark   :", spark.version)

# a tiny computation
df = spark.range(5)
print("Count   :", df.count())

spark.stop()
print("SUCCESS ✅")
```

**Expected output:**

```
Python  : ...\envs\spark\python.exe
PySpark : 3.5.8
Spark   : 3.5.8
Count   : 5
SUCCESS ✅
```

The first time you build a `SparkSession` it takes 10–20 seconds to start the Java engine — that's normal. If you see `SUCCESS ✅`, **you are done!** 🎉

---

## Troubleshooting

### ❌ "Select Kernel" doesn't show *Python (spark) 3.5.8*
- Make sure you ran the `ipykernel install` command in **Part 4** *while `(spark)` was active*.
- In VS Code: **Ctrl+Shift+P → "Developer: Reload Window"**, then click Select Kernel again.
- Confirm the kernel exists: in Anaconda Prompt run `jupyter kernelspec list` — you should see `spark`.

### ❌ `IllegalAccessError ... sun.nio.ch.DirectBuffer`
This means Spark is running on **Java 17 or newer**, which it doesn't support. Fix by forcing Java 11 into the env:
```bash
conda activate spark
conda install -c conda-forge "openjdk=11" -y
```
Then fully restart VS Code.

### ❌ `Py4JException: Constructor org.apache.spark.sql.SparkSession(...) does not exist`
You have a **`SPARK_HOME`** environment variable pointing at a *different* Spark version. Delete it (see the warning box in Part 3), fully restart VS Code, and try again. With the pip install, you should have **no** `SPARK_HOME` at all.

### ❌ (Windows) `Could not locate executable ...winutils.exe` / errors when saving files
You skipped or misconfigured **Part 3**. Recheck that `C:\hadoop\bin\winutils.exe` and `hadoop.dll` exist and that `HADOOP_HOME=C:\hadoop`. Fully restart VS Code afterward.

### ❌ The kernel keeps "connecting" and then dies / never runs a cell
- Selecting a kernel does **not** start it — you must **run a cell** (Shift+Enter) to boot it.
- If it still fails, run the cell once and read the red error text — it usually names the problem (often Java or `winutils`).

### ❌ `command not found: conda`
Close the terminal and open a **new** one (Windows: use **Anaconda Prompt**, not Command Prompt).

---

## Quick reference (cheat sheet)

```bash
# 1. Create the environment (Python + Java 11 + kernel support)
conda create -n spark -c conda-forge python=3.11 openjdk=11 ipykernel -y

# 2. Activate it
conda activate spark

# 3. Install Spark 3.5.8
pip install pyspark==3.5.8

# 4. Register the VS Code / Jupyter kernel
python -m ipykernel install --user --name spark --display-name "Python (spark) 3.5.8"

# 5. (Windows only) put winutils.exe + hadoop.dll in C:\hadoop\bin
#    and set HADOOP_HOME=C:\hadoop  (do NOT set SPARK_HOME)

# 6. In VS Code: Select Kernel -> Python (spark) 3.5.8 -> run a cell
```

**Golden rules**
- ✅ Java **11** (not 17+).
- ✅ Python **3.11** (not 3.13+).
- ✅ `pyspark==3.5.8` via pip.
- 🚫 **Never** set `SPARK_HOME` when using the pip install.
- 🔁 After changing environment variables, **fully quit and reopen** VS Code.

---

*If you're stuck, copy the full red error message and bring it to office hours or the course forum — the exact text almost always points to the fix.*
