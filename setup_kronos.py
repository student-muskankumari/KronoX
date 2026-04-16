"""
Kronos Setup Script — run once.
    python setup_kronos.py

The Kronos repo uses a model/ package (model/__init__.py, model/kronos.py)
not a flat model.py file. This script finds the package root and saves
the path to src/kronos_path.json so the app can use it.
"""
import subprocess, sys, os, shutil, json, importlib.util, importlib.util


def run(cmd, **kw):
    return subprocess.run(cmd, **kw)


def find_kronos_root(base):
    """
    Search for the directory that contains the 'model' package with
    KronosPredictor — i.e. a folder that has model/__init__.py AND
    importing 'from model import KronosPredictor' works.
    Returns (kronos_root, init_path) or (None, None).
    """
    for dirpath, dirnames, _ in os.walk(base):
        if "model" in dirnames:
            init_py = os.path.join(dirpath, "model", "__init__.py")
            if os.path.exists(init_py):
                try:
                    with open(init_py, encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    if "KronosPredictor" in txt or "Kronos" in txt:
                        return dirpath, init_py
                except Exception:
                    pass
    return None, None


def setup():
    base    = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base, "src")
    kdir    = os.path.join(src_dir, "kronos")
    cfg     = os.path.join(src_dir, "kronos_path.json")
    os.makedirs(src_dir, exist_ok=True)

    # ── Already configured? ───────────────────────────────────
    if os.path.exists(cfg):
        with open(cfg) as f:
            info = json.load(f)
        kr = info.get("kronos_root")
        if kr and os.path.exists(os.path.join(kr, "model", "__init__.py")):
            print(f"✅  Already configured — kronos root:\n   {kr}")
            return True

    print("=" * 60)
    print("Kronos Setup")
    print("=" * 60)

    # ── 1. Clone ──────────────────────────────────────────────
    print("\n[1/4] Cloning Kronos repository …")
    if not os.path.exists(kdir):
        tmp = kdir + "_tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        r = run(["git", "clone", "--depth", "1",
                 "https://github.com/shiyu-coder/Kronos", tmp],
                capture_output=True, text=True)
        if r.returncode != 0:
            print(f"❌  git clone failed:\n{r.stderr}")
            print("    Install git: https://git-scm.com/download/win")
            return False
        shutil.move(tmp, kdir)
        print("   ✓  Cloned to src/kronos/")
    else:
        print("   ✓  src/kronos/ already exists (skipping clone)")

    # ── 2. Find model package ─────────────────────────────────
    print("\n[2/4] Locating model package (model/__init__.py) …")
    kronos_root, init_path = find_kronos_root(kdir)

    if kronos_root is None:
        # Fallback: src/kronos/ itself if model/ is directly inside
        candidate = os.path.join(kdir, "model", "__init__.py")
        if os.path.exists(candidate):
            kronos_root = kdir
            init_path   = candidate

    if kronos_root is None:
        print("❌  Could not find model/__init__.py with KronosPredictor.")
        print("    Contents of src/kronos/:")
        for r,d,f in os.walk(kdir):
            for fn in f:
                rel = os.path.relpath(os.path.join(r,fn), kdir)
                if not rel.startswith(".git"):
                    print(f"     {rel}")
            # Stop after first level
            break
        return False

    rel = os.path.relpath(kronos_root, base)
    print(f"   ✓  Found model package at: {rel}/model/")

    # ── 3. Dependencies ───────────────────────────────────────
    req = os.path.join(kdir, "requirements.txt")
    if os.path.exists(req):
        print("\n[3/4] Installing dependencies …")
        r2 = run([sys.executable, "-m", "pip", "install",
                  "-r", req, "--quiet"],
                 capture_output=True, text=True)
        print("   ✓  Done" if r2.returncode == 0
              else f"   ⚠️  Some deps failed (non-fatal):\n{r2.stderr[:300]}")
    else:
        print("\n[3/4] Installing core deps …")
        run([sys.executable, "-m", "pip", "install",
             "einops", "rotary-embedding-torch", "--quiet"])
        print("   ✓  Done")

    # ── 4. Verify ─────────────────────────────────────────────
    print("\n[4/4] Verifying import …")
    if kronos_root not in sys.path:
        sys.path.insert(0, kronos_root)
    try:
        # Import the model package
        import importlib
        spec = importlib.util.spec_from_file_location(
            "_kronos_model_pkg",
            init_path,
            submodule_search_locations=[os.path.dirname(init_path)])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "KronosPredictor") or hasattr(mod, "Kronos"), \
            "KronosPredictor/Kronos not in model/__init__.py"
        print("   ✓  model package verified")
    except Exception as e:
        print(f"   ⚠️  Verification warning (non-fatal): {e}")
        print("      Will attempt runtime import when app starts.")

    # Save config
    with open(cfg, "w") as f:
        json.dump({
            "kronos_root": kronos_root,
            "model_init":  init_path,
        }, f, indent=2)
    print(f"   ✓  Config saved to src/kronos_path.json")

    print("\n" + "=" * 60)
    print("✅  Setup complete!")
    print("   Model weights (~50 MB) download on first prediction.")
    print("   Restart app:  streamlit run app_hourly.py")
    print("=" * 60)
    return True


if __name__ == "__main__":
    sys.exit(0 if setup() else 1)
