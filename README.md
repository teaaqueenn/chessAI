# ♟️ AI Chess with Reinforcement Learning

TLDR; Built a reinforcement-learning chess AI from scratch in Python, trained on 700+ games with custom reward functions; visualized performance using precision, recall, and F1 graphs. Note: Most complete versions live in pre-trained-edition, saving, or v7-touchscreen-version branches; repo has messy branches due to early Git mistakes.

organization = refactor without graphs

stockfish-rewards = Stockfish integration experiment

Photos of one of the graphs + AI Vs Human
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6ac221ca-9806-4dc1-90ff-b2197cafc3e7" />
<img width="1920" height="1069" alt="image" src="https://github.com/user-attachments/assets/d70e039e-8733-420f-ab65-d241dbdfce45" />
More graphs are in Chess/graphs in a previously mentioned branch.



## Overview

This project is a self-driven exploration into reinforcement learning, applied to chess. The goal was to build an AI agent that learns how to play chess through self-play and training, without prior domain knowledge. Initially motivated by the fact that all my friends knew how to play chess and I didn’t, this turned into a full development and learning experience in machine learning, game logic, and performance visualization.

The project is built in Python and evolved through multiple iterations, learning from failures and making gradual improvements — from scratch-built rule logic to AI integration and performance tracking through graphs and custom metrics.

The branches on this project are MESSY, but I tried. "organization" is the branch where i restructured the code to be cleaner, however, in the process I did not implement live graphing utility. "stockfish-rewards" was my attempt to utilize stockfish's API as the rewards system instead of my custom rewards system. either "pre-trained-edition", "saving" or "v7-touchscreen-version" is the most complete version without the experiments mentioned above. I made this 8 months ago and my branch naming was horrible (as were my git conventions), forgive me. There are graphs for training and for when one of my friends (who actually knows how to play chess) played against it. They are in the organization branch, along with a few others.

---

## 🚧 Development Timeline

### Version 1
- Implemented chess rules from scratch.
- System broke when handling special rules like en passant and castling.

### Version 2
- Added AI too early without a solid base.
- Resulted in a broken build.

### Version 3
- Built a text-based board UI.
- Poor visual feedback and failed board initialization.

### Version 4
- Experimental build. Unstable and unrecoverable.

### Version 5 / 5.1
- Attempted updates without proper version control.
- Overwrote files and lost progress.

### Version 6
- Introduced GitHub for version control.
- Successfully created a functional baseline.

### Version 7
- Fully playable version with multiple game modes.
- Stable and used as the foundation for all future work.

### Version 7.1
- Integrated AI vs AI gameplay (fully working).
- Fixed reward system not updating properly.
- Graphing initially failed due to incorrect imports; later resolved.
- Added touchscreen support (resolved complex indexing issues).
- Added and visualized multiple metrics and training graphs.
- Developed over 20 custom functions for expanded reward mechanics.
- Passed 1000 lines of code.
- Implemented saving/loading to avoid excessive retraining.
- Trained on 700 games. Learning is slow but consistent.

---

## 📈 Training Metrics & Graphs

The model tracks and visualizes learning performance using the following metrics:

1. **Training Loss Curve**  
   *Shows how far the model's predictions deviate from expected outcomes.*  
   → Goal: decrease over time.

2. **Accuracy Over Epochs**  
   *Tracks how often predictions are correct during training.*  
   → Goal: increase steadily.

3. **Precision Over Epochs**  
   *Measures how many predicted moves were actually correct.*  
   → High precision = fewer false positives.

4. **Recall Over Epochs**  
   *Measures how many actual good moves were correctly predicted.*  
   → High recall = model captures most relevant options.

5. **F1 Score Over Epochs**  
   *Balance between precision and recall.*  
   → Higher = more consistent and complete decision-making.

6. **Gradient Plot**  
   *Visualizes how much the model is adjusting weights during learning.*  
   → Helps identify vanishing or exploding gradients.

7. **Output Probability Distribution**  
   *Represents model’s confidence in its decisions.*  
   → Sharp peaks = strong predictions.

8. **Weight Histograms**  
   *Tracks internal model weight values.*  
   → Balanced weights indicate healthy training.

9. **Confusion Matrix**  
   *Displays which predictions were right or wrong.*  
   → Useful for spotting recurring misclassifications.

10. **Custom Metric Curve**  
   *Monitors improvements in selected metrics (e.g., precision/F1).*  
   → Helps with targeted tuning.

---

## 🧪 Final Testing

After model training, testing was conducted against a human player (a friend who actually knows chess). The results showed gradual and consistent improvement over time, verified using Q-value trend graphs. While the model is still beatable, it demonstrates real learning capability through reinforcement.

---

## ✅ Current Status

- Reinforcement learning works
- AI improves with more games
- Graphs and metrics show meaningful trends
- Still requires more training for competitive play

---

## 📌 Future Plans

- Accelerate training cycles with better hardware or parallelization
- Expand reward system using deeper strategic evaluation
- Refactor codebase for modularity and scalability
- Improve interface and visualization (potential GUI)
- Publish trained models for reproducibility

---

## 🧠 Lessons Learned

- Always use version control (GitHub saved this project!)
- Start with the fundamentals before introducing AI
- Visualizing model behavior is essential to debugging and improving
- Reinforcement learning is powerful — and incredibly slow without the right setup

---

## 💻 Tech Stack

- **Language**: Python
- **Learning Framework**: Custom Reinforcement Learning Loop
- **Visualization**: Matplotlib
- **Version Control**: GitHub

---
