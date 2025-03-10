# Tetris AI Project

## **Demo Video**
[![Tetris AI Demo](https://img.youtube.com/vi/D8MjBG5kSzU/0.jpg)](https://www.youtube.com/watch?v=D8MjBG5kSzU)

## **Tetris AI in Action**
![Tetris AI Playing](assets/tetris_ai_demo.gif)

## **Genetic Algorithm in Action**
![Tetris AI Playing](assets/ga_ai_demo.gif)

## Overview
This project is an AI-driven Tetris player built using **Python** and **Pygame**. It leverages **Deep Q-Networks (DQN), Double DQN, Prioritized Experience Replay, and Genetic Algorithms** to train an agent that can efficiently play Tetris. The project underwent significant optimizations from **Version 1** to **Version 2** to enhance training speed and efficiency.

## Environment
The game environment follows **NES Tetris** rules, implementing:
- **Scoring system** similar to NES Tetris.
- **Gravity mechanics** for line clears.

The AI interacts with the game through **state-based decisions**, selecting moves from all possible placements and rotations.

## AI Agent
The initial AI agent was based on **DQN**, with a **single network** estimating both current and target Q-values. However, this led to **overestimations and early convergence**. To address this, we switched to **Double DQN**, where:
- The **primary network** predicts actions and updates every 200 pieces.
- The **target network** stabilizes learning by updating every 1000 pieces.

### Reward Function Design
Designing a balanced reward function was challenging. If the agent was **only rewarded for clearing lines**, it struggled to learn the steps required to set up efficient line clears. Instead, we introduced **sparse rewards** that encourage moves leading to a **better board state**.

- A **good board** should have:
  - **Minimal bumpiness** (smooth surfaces for easier clears)
  - **Minimal holes** (no trapped empty spaces)
  - **Small pillars** (avoiding difficult-to-clear structures)

To achieve this:
- The agent was **punished for increasing bumpiness, holes, or large pillars**.
- **Stacking too high resulted in additional penalties**.
- Moves that **created stable board structures** were **encouraged**.

This approach helped the agent develop **long-term strategies** rather than focusing only on immediate rewards.

### Exploration vs. Exploitation Strategy
Instead of relying solely on a **typical decay schedule**, we combined it with an **alternating strategy** between **high exploration and high exploitation** in **500-game cycles**. This method **sped up learning while maintaining stability**.

#### High Exploration Phase (500 games)
- **Epsilon:** `0.3 → 0.0001`
- **Learning Rate (LR):** `0.01 → 0.001`
- Since the agent has **10-40 move choices per state**, high exploration **encourages broader strategy discovery**.
- A **higher learning rate (LR)** allows more aggressive updates, helping the agent learn **board setup strategies faster**.

#### High Exploitation Phase (500 games)
- **Epsilon:** `0.0001`
- **Learning Rate (LR):** `0.001`
- The agent **tests its learned strategies** from the exploration phase.
- **Lower LR prevents drastic updates**, refining the strategy **without overfitting**.
- This phase **stabilizes** the agent's learning, similar to how **stocks correct after a surge**.

#### Second Cycle of Exploration & Exploitation
- In the first cycle, exploration happened **without prior knowledge**.
- In the second cycle, the agent **explored with refined strategies**, leading to more **targeted discoveries**.
- Another **500-game exploration phase** allowed for additional improvements.
- The final **exploitation phase fine-tuned an even better strategy**.
- **One full cycle** was enough for a **high-scoring AI (1M+ points)**.
- **Two cycles further optimized** the agent, making it even stronger.

This alternating method helped the agent **learn, refine, explore deeper, and perfect its strategy**.

### Neural Network Architecture
- **Input Size**: 6 features (total_height, bumpiness, holes, line_cleared, y_pos, pillar)
- **Hidden Layers**: 2 layers, **32 neurons each**
- **Output**: 1 action per placement
- **Learning Rate**: Starts at `0.01`, decaying to `0.001`
- **Gamma (Discount Factor):** `0.999`
- **Exploration Strategy**: Epsilon-greedy decay from `0.3 → 0.0001`
- **Batch Size**: `128`
- **Training Epochs**: `2 per iteration`

### Prioritized Experience Replay
Instead of randomly selecting past experiences, we implemented **prioritized replay**, where experiences are chosen based on **TD error** (difference between predicted and actual Q-values). This **significantly improved early learning efficiency**.

## Genetic Algorithm (GA)
To evolve better-performing agents, we implemented a **Genetic Algorithm (GA)** that allowed multiple agents to train simultaneously.

- **Version 1:** The project was **not originally designed** to handle multiple game boards in one window. As a workaround, we used **multiprocessing**, giving each agent its **own CPU core**. However, this approach **limited us to 10 agents**, constrained by available CPU processors.  

- **Version 2:** Knowing we wanted **many agents running at once**, we **redesigned the project** to support multiple boards within a single process. This **eliminated the need for multiprocessing**, allowing the computer to efficiently manage tasks internally. Thanks to optimizations, we increased the number of agents from **10 to 250**.

This major redesign made **GA training far more scalable**, allowing **larger populations** and better evolution.

## Optimizations (Version 1 → Version 2)
Profiling revealed **two major bottlenecks**:
1. **Rendering inefficiencies** - Redrawing static elements every frame.
2. **State calculation overhead** - Dropping pieces in all possible positions consumed excessive time.

### Rendering Optimizations
- **Old Approach**: Redrew **every block** in every frame.
- **New Approach**: Used **dirty rects** (only updating changed areas).  
  **Result**: Rendering time reduced from **90s → 5s**.

### State Calculation Optimizations
- **Old Approach**: Used Python loops, making `calc_all_states()` slow (**~180s**).
- **New Approach**: Rewrote with **Numba’s njit** for machine code execution.  
  **Result**: Execution time reduced from **160s → 25s**.

### Additional Optimizations
- **Blitting Optimization**: Rendered directly to the main screen instead of intermediate surfaces.
- **Batch Processing**: Consolidated multiple small calculations into fewer large ones.
- **Reduced Redundant Board Operations**: Minimized unnecessary board evaluations.

---

### **Version 1 Profiling (500 games)**
```plaintext
223807016 function calls (210035110 primitive calls) in 483.520 seconds

Ordered by: cumulative time

ncalls tottime percall cumtime percall filename:lineno(function) 

1        2.155    2.155  481.741  481.741 train.py:85(run_simulation) 
42254    0.305    0.000  206.318    0.005 tetris.py:87(play_full) 
82161    0.575    0.000  205.107    0.002 tetris.py:139(play_step) 
42383   12.129    0.000  180.555    0.004 game.py:203(calc_all_states) 
82160    0.253    0.000  114.913    0.001 game.py:250(run) 
1005317  4.698    0.000  96.764     0.000 game.py:311(hard_drop) 
10562398 16.251   0.000  92.066     0.000 game.py:316(move_down)
```

### **Version 2 Profiling (500 games)**
```plaintext
22190082 function calls (20214157 primitive calls) in 52.530 seconds

Ordered by: cumulative time

ncalls tottime percall cumtime percall filename:lineno(function) 

1        0.423    0.423   50.491   50.491 main_screen.py:155(run2) 
46619    3.519    0.000   20.576    0.000 main_screen.py:113(play_action) 
31/21    0.000    0.000   17.968    0.856 _ops.py:291(fallthrough) 
698733/140673  0.902    0.000   17.785    0.000 module.py:1735(_wrapped_call_impl) 
698733/140673  1.156    0.000   17.579    0.000 module.py:1743(_call_impl) 
139515   1.450    0.000   17.034    0.000 model.py:12(forward)
```

### **Key Takeaways**
- **Total runtime reduced from 483.52s → 52.53s (≈89% speedup)**
- **`calc_all_states()` reduced from 180s → ~5s**
- **Rendering reduced from 90s → ~5s**
- **Overall, training is significantly faster and more scalable.**

---

## **Running the Project**
To run the AI, navigate to the appropriate version and execute:

### **Install requirements**
```bash
pip install -r requirements.txt
```

### **Version 1**
```bash
cd version1
python train.py
```

### **Version 2**
```bash
cd version2
python main_screen.py
```

