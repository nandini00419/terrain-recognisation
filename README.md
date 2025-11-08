# 🌍 Deep Learning Terrain Recognition System

A deep learning web application that classifies terrain types (desert, forest, mountain, plain, urban, water) from images using Convolutional Neural Networks (CNN) built with TensorFlow/Keras and Flask.

## 🎯 Features

- **Deep Learning Model**: Uses MobileNetV2 (transfer learning) for efficient terrain classification
- **Interactive Dashboard**: Beautiful dashboard with statistics, charts, and visualizations
- **Web Interface**: Clean, modern Flask web application with drag-and-drop image upload
- **Real-time Predictions**: Get instant terrain classification results with confidence scores
- **Prediction History**: Track all your predictions with search and filter capabilities
- **Data Visualizations**: Interactive charts showing terrain distribution and confidence metrics
- **AI Cursor Feature**: Interactive cursor that shows terrain predictions as you hover over images
- **Top 3 Predictions**: View the top 3 most likely terrain types with confidence percentages
- **Animated UI**: Smooth animations and transitions throughout the application
- **Model Performance Metrics**: View training and validation accuracy/loss metrics
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Requirements

- Python 3.10+
- TensorFlow 2.15+
- Flask 3.0+
- Dataset organized in folders by terrain type

## 🚀 Installation

1. **Clone the repository**
   ```bash
   cd terrain-recognisation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   Organize your images in the `_dataset` folder with the following structure:
   ```
   _dataset/
   ├── desert/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── forest/
   │   ├── image1.jpg
   │   └── ...
   ├── mountain/
   ├── plain/
   ├── urban/
   └── water/
   ```

## 🏋️ Training the Model

Train the terrain recognition model using the training script:

```bash
python train_model.py
```

The script will:
- Load and preprocess images from `_dataset/`
- Apply data augmentation (rotation, zoom, flip, etc.)
- Split data into training (80%) and validation (20%) sets
- Train a MobileNetV2-based CNN model
- Save the trained model as `terrain_model.h5`
- Generate training history plots

**Training Parameters** (can be modified in `train_model.py`):
- Image Size: 224x224 pixels
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.001
- Model: MobileNetV2 (transfer learning)

## 🌐 Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload an image**
   - Click the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Maximum file size: 16MB

4. **View predictions**
   - The model will predict the terrain type
   - View top 3 predictions with confidence scores
   - Hover over the image to see AI cursor predictions (if enabled)

5. **Explore Dashboard**
   - Navigate to `/dashboard` to see statistics and visualizations
   - View terrain distribution charts
   - Check model performance metrics
   - See recent predictions

6. **View History**
   - Navigate to `/history` to see all past predictions
   - Search and filter predictions by terrain type
   - View prediction details and confidence scores

## 📁 Project Structure

```
terrain-recognisation/
├── _dataset/              # Training images (organized by terrain type)
├── templates/             # HTML templates
│   ├── index.html        # Upload page
│   ├── result.html       # Results page
│   ├── dashboard.html    # Dashboard with statistics
│   └── history.html      # Prediction history page
├── static/               # Static files
│   ├── css/
│   │   └── style.css    # Stylesheet with animations
│   ├── js/
│   │   ├── main.js      # Upload page JavaScript
│   │   ├── result.js    # Results page JavaScript (AI cursor)
│   │   ├── dashboard.js # Dashboard JavaScript (charts)
│   │   └── history.js   # History page JavaScript
│   └── uploads/         # Uploaded images storage
├── train_model.py        # Model training script
├── app.py                # Flask web application (with dashboard routes)
├── terrain_model.h5      # Trained model (generated after training)
├── class_indices.json    # Class name mapping (generated after training)
├── prediction_history.json # Prediction history (generated)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── GITHUB_SETUP.md      # GitHub setup instructions
```

## 🎨 Model Architecture

The model uses **MobileNetV2** as a base (transfer learning) with a custom classification head:

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Global Average Pooling**: Reduces spatial dimensions
- **Dropout Layers**: Prevents overfitting (0.3 dropout rate)
- **Dense Layers**: 128 units with ReLU activation
- **Output Layer**: Softmax activation for multi-class classification

## 🔧 Customization

### Adding New Terrain Types

1. Create a new folder in `_dataset/` with your terrain name
2. Add training images to the folder
3. The model will automatically detect the new class during training
4. Update the `TERRAIN_CLASSES` list in `train_model.py` if needed

### Modifying Model Architecture

Edit `train_model.py` to customize:
- Model architecture (switch to VGG16, ResNet, etc.)
- Training parameters (epochs, batch size, learning rate)
- Data augmentation options
- Image input size

### Styling

Modify `static/css/style.css` to customize the web interface appearance.

## 📊 Model Evaluation

After training, the model displays:
- Training and validation accuracy curves
- Training and validation loss curves
- Final validation accuracy and loss

Check `training_history.png` for visual training metrics.

## 🐛 Troubleshooting

### Model Not Found Error
- Make sure you've trained the model first: `python train_model.py`
- Ensure `terrain_model.h5` exists in the project root

### Dataset Not Found
- Verify your `_dataset/` folder exists
- Check that images are organized in subdirectories by terrain type

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.10+ required)

### Memory Errors During Training
- Reduce batch size in `train_model.py`
- Reduce image size (IMG_SIZE)
- Use fewer training images

## 🚀 Deployment

### Local Deployment
The Flask app runs on `http://localhost:5000` by default.

### Production Deployment
For production, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up a reverse proxy (Nginx)
- Using environment variables for configuration
- Implementing proper error handling and logging
- Adding authentication/authorization if needed

## 📊 Dashboard Features

The interactive dashboard provides:

- **Statistics Cards**: Total predictions, model accuracy, terrain classes, average confidence
- **Terrain Distribution Chart**: Interactive doughnut chart showing distribution of predicted terrains
- **Confidence Distribution Chart**: Bar chart showing confidence score distribution
- **Model Performance Metrics**: Training and validation accuracy/loss with animated progress bars
- **Recent Predictions**: Last 5 predictions with details
- **Real-time Updates**: Charts update every 30 seconds

## 📜 Prediction History

The history page allows you to:

- View all past predictions with images
- Search predictions by terrain type
- Filter by terrain class
- See prediction timestamps and confidence scores
- Navigate through prediction history

## 🎨 UI/UX Features

- **Smooth Animations**: Fade-in, slide-up, and pulse animations
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Interactive Charts**: Powered by Chart.js with hover effects
- **Loading States**: Animated loading spinners and progress indicators
- **Color-coded UI**: Intuitive color scheme for different terrain types

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Flask for the web framework
- MobileNetV2 pre-trained model from TensorFlow Hub
- Chart.js for data visualizations

## 📧 Support

For issues or questions, please open an issue on the repository.

## 🔗 GitHub

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for instructions on setting up the GitHub repository.

---

**Happy Terrain Recognizing! 🌍✨**

