# 💎 Jewelry Image Generator & Fine-Tuner - Instructions

## Introduction

Welcome! This tool allows you to:

1.  **Generate** unique jewelry images using pre-trained AI models.
2.  **Fine-tune** the AI model on your own specific jewelry images (using LoRA technology) to create customized styles.
3.  **Manage** different jewelry styles or client projects in separate folders.

This guide will walk you through setting up and using the application.

## Prerequisites (What You Need)

1.  **Computer:**
    *   A reasonably modern computer (Windows, macOS, or Linux).
    *   **For Fine-Tuning:** A powerful **NVIDIA GPU** (like RTX 20xx, 30xx, 40xx series or professional equivalents like T4, A10G) with **at least 8-12GB of VRAM** is *highly recommended*. Fine-tuning on CPU is possible but extremely slow (days instead of hours). Generation can work on CPU but will also be much slower than on GPU.
    *   **Disk Space:** At least **20-30 GB** of free space for Python, libraries, the base AI model, the training script repository, and your generated images/fine-tuned models.
    *   **Internet:** Needed for initial download of libraries and models.
2.  **Software:**
    *   **Python:** Version 3.9, 3.10, or 3.11 recommended. ([Download Python](https://www.python.org/downloads/))
    *   **Git:** Needed to download the base model and training scripts. ([Download Git](https://git-scm.com/downloads/))

## Setup Steps (One-Time Only)

Follow these steps carefully to prepare your system.

**Step 1: Create a Project Folder**

*   Create a new folder on your computer where you will store the application code, models, and your jewelry images. Let's call it `JewelryAI`.
*   All subsequent commands should be run *inside* this `JewelryAI` folder using a terminal or command prompt.

**Step 2: Set Up Python Environment**

*   Open your terminal or command prompt (like Terminal on macOS/Linux, Command Prompt or PowerShell on Windows).
*   Navigate *into* your `JewelryAI` folder using the `cd` command. Example: `cd path/to/your/JewelryAI`
*   **Create a Virtual Environment (Recommended):** This keeps the project's libraries separate from your system's Python.
    ```
    python -m venv env
    ```
*   **Activate the Virtual Environment:**
    *   **Windows (Command Prompt):** `env\Scripts\activate.bat`
    *   **Windows (PowerShell):** `env\Scripts\Activate.ps1` (You might need to set execution policy: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first)
    *   **macOS / Linux:** `source env/bin/activate`
    *   You should see `(env)` appear at the beginning of your terminal prompt. Keep this terminal open and active for the following steps.

**Step 3: Download Application Code**

*   Save the Python script provided (containing the Streamlit application logic) as `app.py` inside your `JewelryAI` folder.
*   Save the `requirements.txt` file (provided above) into the same `JewelryAI` folder.

**Step 4: Install Dependencies (Libraries)**

*   **Crucial - Install PyTorch (GPU or CPU):**
    *   Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your Operating System, Package (`Pip`), Language (`Python`), and Compute Platform (`CUDA` version if you have NVIDIA GPU, or `CPU`).
    *   Copy the generated installation command (it will look something like `pip3 install torch torchvision torchaudio --index-url ...` for CUDA, or `pip3 install torch torchvision torchaudio` for CPU).
    *   Run **that specific command** in your activated `(env)` terminal. This ensures you get the correct version for your hardware.
*   **Install Other Libraries:** Once PyTorch is installed, install the rest using the `requirements.txt` file:
    ```
    pip install -r requirements.txt
    ```
    *   *(Note: `bitsandbytes` might fail on Windows/macOS if not using WSL. The app will still work, but the "8-bit Adam" option during fine-tuning won't be effective.)*
*   **Configure `accelerate`:** This library helps launch the training. Run its configuration command:
    ```
    accelerate config
    ```
    *   Answer the questions. For typical local use:
        *   Compute environment: `This machine` (0)
        *   Distributed type: `No` (usually 0)
        *   (Answer No if asked about DeepSpeed/FP8)
        *   GPU selection: Choose `all` if you have one GPU, or specify IDs if needed.
        *   Mixed precision: Choose `fp16` or `bf16` if your GPU supports it (recommended), otherwise `no`.

**Step 5: Download Base AI Model (Stable Diffusion v1.5)**

*   This model is the foundation. We download a copy locally. Run this command in your `(env)` terminal:
    ```
    git clone https://huggingface.co/sd-legacy/stable-diffusion-v1-5 base_models/stable-diffusion-v1-5
    ```
*   **Check for Errors:** Watch the output carefully. Make sure it completes **without** any `checkout failed` or other critical errors. If you see errors, try cloning into a simpler path or check disk space/permissions.
*   This will create a folder structure like: `JewelryAI/base_models/stable-diffusion-v1-5/` containing model files.

**Step 6: Download Training Script Repository (Diffusers)**

*   This contains the script needed for fine-tuning. Run this command in your `(env)` terminal:
    ```
    git clone https://github.com/huggingface/diffusers.git diffusers_repo
    ```
*   This creates `JewelryAI/diffusers_repo/` containing the necessary code.

**Setup Complete!** You shouldn't need to repeat these steps unless you move to a new computer or need to update libraries.

## Running the Application

1.  **Activate Environment:** Open a new terminal, navigate to your `JewelryAI` folder, and activate the virtual environment:
    *   Windows: `env\Scripts\activate.bat`
    *   macOS/Linux: `source env/bin/activate`
2.  **Run Streamlit:** Start the application using:
    ```
    streamlit run app.py
    ```
3.  **Access in Browser:** Your web browser should automatically open to the application's interface (usually `http://localhost:8501`).

## Using the Application (Workflow)

The application interface has a sidebar for setup and main area with tabs for different actions.

**1. Create Client & Product Folders**

*   **Use the Sidebar:**
    *   Enter a unique name for a "Client" (e.g., `ClientA`, `MyDesigns`) and click "Create".
    *   Select the newly created Client from the dropdown.
    *   Enter a unique name for a "Product" under that client (e.g., `Gold_Rings`, `Silver_Necklaces_Style1`) and click "Create".
*   **What happens:** This creates folders automatically:
    *   `JewelryAI/inputs/YourClient/YourProduct/` (For your training images)
    *   `JewelryAI/inputs/YourClient/YourProduct/model/` (Where fine-tuned weights will be saved)
    *   `JewelryAI/outputs/YourClient/YourProduct/` (Where generated images will be saved)

**2. Add Training Images (for Fine-Tuning)**

*   **Find the Folder:** Locate the `JewelryAI/inputs/YourClient/YourProduct/` folder on your computer.
*   **Copy Images:** Place your training images (e.g., `.jpg`, `.png`, `.webp` files) **directly** into this folder. Do *not* put them in the `model` subfolder.
*   **Quality & Quantity:** Use clear, well-lit images representative of the style you want. Aim for at least 10-20 high-quality images, but more is often better (up to a few hundred).

**3. Fine-Tune Model (Optional - Requires GPU!)**

*   **Select:** Make sure the correct Client and Product are selected in the sidebar.
*   **Go to Tab:** Click the **'⚙️ Fine-Tune Model (LoRA)'** tab.
*   **Verify Images:** You should see a preview of the images you added in Step 2. If not, check you put them in the right folder.
*   **Adjust Parameters (Optional):**
    *   **Max Training Steps:** Important! Higher means longer training. Start with 500-1500 depending on image count. Too high can "burn" the style.
    *   **Learning Rate:** Often okay to leave at default (e.g., `1e-4`).
    *   **LoRA Rank:** Controls complexity. 8-16 is usually a good start.
    *   (Leave advanced options as default unless you know what they do).
*   **Start:** Click the **'🚀 Start Fine-Tuning'** button.
*   **Wait:** Monitor the "Training Logs" expander. This will take time (potentially hours on consumer GPUs). **Do not close the terminal or the browser tab.**
*   **Completion:** If successful, it will say "Training process completed successfully!" and the custom LoRA file (`pytorch_lora_weights.safetensors`) will be saved in the `JewelryAI/inputs/YourClient/YourProduct/model/` folder.

**4. Generate Images**

*   **Select:** Choose the Client and Product in the sidebar. If you just finished fine-tuning, re-selecting the product ensures the new LoRA weights are loaded.
*   **Go to Tab:** Click the **'✨ Generate Image'** tab.
*   **Click Generate:** Simply click the **'Generate Image'** button.
    *   The app uses a fixed prompt: `"a beautiful piece of jewelry"`.
    *   If a fine-tuned LoRA model exists for the selected product (in its `model` folder), it will be automatically applied to influence the style. Otherwise, it uses the base Stable Diffusion model.
*   **View:** The generated image appears below the button. It's also saved in the `JewelryAI/outputs/YourClient/YourProduct/` folder.

**5. View & Download Designs**

*   **Select:** Choose the Client and Product in the sidebar.
*   **Go to Tab:** Click the **'🖼️ Generated Designs'** tab.
*   **Browse:** See all images generated for this product.
*   **Download:** Use the "Download All ... Images (.zip)" button to save a zip file containing all images from that product's output folder.

## Simple Example Scenario: Fine-Tuning "Gold Rings"

1.  **Run App:** `streamlit run app.py`
2.  **Create:** In sidebar, create Client `MyJewelry`, then select it. Create Product `Gold_Rings`.
3.  **Add Images:** Find `JewelryAI/inputs/MyJewelry/Gold_Rings/` on your computer. Copy 15 `.png` photos of gold rings into this folder.
4.  **Fine-Tune:**
    *   Go to '⚙️ Fine-Tune' tab. Verify the 15 images appear.
    *   Set `Max Training Steps` maybe to `800` (adjust based on how many epochs you estimate).
    *   Leave other settings at default for now.
    *   Click '🚀 Start Fine-Tuning'. Wait (e.g., 30-60 minutes, depends heavily on GPU).
    *   Wait for success message. The `pytorch_lora_weights.safetensors` file is now in `.../Gold_Rings/model/`.
5.  **Generate:**
    *   Go to '✨ Generate Image' tab. (Ensure `MyJewelry` / `Gold_Rings` is selected in sidebar - re-select if needed).
    *   Click 'Generate Image'. The output should reflect the style of your training images.
    *   Click a few more times to get variations.
6.  **View:** Go to '🖼️ Generated Designs' tab to see all generated gold ring images. Use the download button if desired.

## Troubleshooting

*   **`ModuleNotFoundError`:** You missed installing a library. Activate `(env)` and run `pip install -r requirements.txt` again, or `pip install <missing_library_name>`. Check PyTorch installation specifically.
*   **`git` not found:** Install Git from [https://git-scm.com/downloads/](https://git-scm.com/downloads/).
*   **`accelerate` not found / `accelerate config` fails:** Ensure accelerate is installed (`pip show accelerate`). Try running `accelerate config` again.
*   **Base Model Errors (`checkout failed`, `model_index.json not found`, `.bin not found`):** The base model download likely failed. Delete the `JewelryAI/base_models/stable-diffusion-v1-5` folder and run the `git clone` command again carefully, checking for errors. Try cloning to a simpler path if necessary.
*   **Fine-Tuning Fails Immediately:** Check `accelerate config`. Ensure `diffusers_repo` exists. Check dependencies (`bitsandbytes` if using 8-bit Adam). Make sure training images exist.
*   **Fine-Tuning Fails Mid-Way (`CUDA out of memory`):** Reduce `Batch Size` (try 1), increase `Gradient Accumulation`, enable `Gradient Checkpointing`, use `fp16` mixed precision, or try lower `Resolution`. Your GPU may not have enough VRAM for the chosen settings.
*   **Generated Images Don't Reflect Fine-Tuning:** Ensure the LoRA file (`pytorch_lora_weights.safetensors`) exists in the correct `.../model/` folder. Re-select the product in the sidebar after fine-tuning to force a reload. Check the LoRA filename matches `LORA_WEIGHT_NAME` in `app.py`.

## Important Notes

*   **Fine-tuning is resource-intensive!** Be patient and ensure you have adequate hardware (especially GPU VRAM).
*   **File Locations Matter:** Pay close attention to where training images go (`inputs/Client/Product/`) versus where LoRA models are saved (`inputs/Client/Product/model/`) and where generated images land (`outputs/Client/Product/`).
*   **Backups:** Regularly back up your `inputs` folder (especially the `model` subfolders containing your fine-tuned LoRAs) and your `outputs` folder.
*   **Experiment:** Try different training parameters (steps, learning rate, rank) to see how they affect the final style. Start with shorter training runs first.

Have fun creating unique jewelry designs! ✨
```
