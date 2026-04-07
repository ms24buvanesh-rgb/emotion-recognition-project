# 🎭 EmoSense — Face Emotion Recognizer

A real-time facial emotion detection web app using TensorFlow, OpenCV, and Flask.

---

## 📋 What You Have

- ✅ **app.py** — Complete Flask backend with emotion detection
- ✅ **index.html** — Beautiful frontend with camera & file upload
- ✅ **requirements.txt** — All Python dependencies

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**⏱️ First time?** This may take 5-10 minutes. TensorFlow downloads are large (~500MB).

### 2. **Run the Server**

```bash
python app.py
```

You should see:

```
✓ Loading TensorFlow model... (this takes ~10-15 seconds on first run)
✓ TensorFlow model loaded successfully
✓ Haar Cascade loaded from: ...
==============================================================================
🚀 EmoSense Server Ready!
==============================================================================

📱 Open your browser to: http://localhost:5000
```

### 3. **Open in Browser**

Navigate to: **http://localhost:5000**

---

## 🎥 How to Use

### **Upload an Image**
1. Click the drop zone or select a file
2. Choose a JPG/PNG with a face
3. Wait for analysis (~2-3 seconds)
4. See emotion + confidence + breakdown

### **Use Your Camera**
1. Click the **"Camera"** tab
2. Click **"Start Camera"**
3. Allow browser permission
4. Click **"Capture Snapshot"**
5. Analysis runs automatically
6. Click **"Retake"** to try again

### **Set Custom Backend (Optional)**
- If you have another emotion detection API, paste its URL in the **API Endpoint** field
- Otherwise, leave blank — the app uses Claude Vision as fallback

---

## 🔧 Troubleshooting

### **"Can't open file app.py: [Errno 2]"**
- **Cause:** app.py not in current directory
- **Fix:** Make sure you're in the folder with `app.py`
  ```bash
  cd C:\Users\acer\Documents  # or wherever your files are
  python app.py
  ```

### **Camera permission denied**
- **Chrome/Edge/Firefox:** Click the permission prompt at the top
- **Safari:** Settings → Privacy → Camera → Allow this site

### **"No module named flask" / "No module named tensorflow"**
- **Fix:** Run the install command again
  ```bash
  pip install -r requirements.txt
  ```
- **Stuck?** Try:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### **Model takes forever to load**
- First run: Normal (~15 seconds while TensorFlow initializes)
- Subsequent runs: ~3-5 seconds
- Patience! ☕

### **"Port 5000 already in use"**
- **Option 1:** Close the other app using port 5000
- **Option 2:** Modify line in `app.py`:
  ```python
  app.run(port=5001)  # Use 5001 instead
  ```

---

## 📊 Emotion Categories

The model recognizes **7 emotions**:

| Emotion | Symbol |
|---------|--------|
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Fear | 😨 |
| Surprise | 😮 |
| Disgust | 🤮 |
| Neutral | 😐 |

---

## ⚙️ Tech Stack

- **Backend:** Flask 2.3+
- **ML Model:** TensorFlow 2.12+ with Keras
- **Face Detection:** OpenCV (Haar Cascade)
- **Image Processing:** Pillow, NumPy
- **Frontend:** Vanilla JS, HTML5, CSS3
- **API Fallback:** Anthropic Claude Vision

---

## 🌐 File Structure

```
your-folder/
├── app.py              # Flask backend
├── index.html          # Web frontend
├── requirements.txt    # Python dependencies
└── SETUP_GUIDE.md      # This file
```

---

## 🔒 Privacy

- ✅ Images processed **locally** (no cloud unless you enable custom API)
- ✅ Camera feed **not recorded** — snapshots deleted after analysis
- ✅ Browser permissions required for camera access

---

## 💡 Advanced Tips

### **Improve Accuracy**
- Good lighting essential
- Face should fill ~30-50% of frame
- Straight-on angle works best
- Avoid shadows/glasses if possible

### **Batch Processing**
- Upload multiple images with drag-and-drop
- Each analyzed independently
- Results appear in history strip

### **API Integration**
Plug in your own emotion detection API:
1. Go to **API Endpoint** field
2. Paste your backend URL (e.g., `http://your-server:8000`)
3. Make sure your API accepts `POST /predict` with form-data `file` key
4. Returns JSON: `{emotion, confidence, all_emotions}`

---

## 📞 Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Verify Flask is installed:**
   ```bash
   python -c "import flask; print(flask.__version__)"
   ```

3. **Check browser console** (F12) for JavaScript errors

4. **Restart** both server and browser

---

## 🎉 You're All Set!

Enjoy emotion detection! 🎭✨

---

**Questions?** Check the console output for error details.
