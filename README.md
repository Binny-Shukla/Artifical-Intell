# 🤖 Artificial-Intell – Redefining Control in Continuous Action Spaces

Hi there! I'm **Brijesh**, currently researching **control systems** 🧠⚙️.  
This project is my take on reinventing **Soft Actor-Critic (SAC)**—a powerful continuous-space RL algorithm—by addressing its key weaknesses in **high-dimensional environments**.  

This implementation has been rigorously tested on one of the hardest OpenAI Gym environments: **Humanoid-v5** 🏃‍♂️💥

---

## 🚀 What's New in This SAC Implementation?

The proposed enhancements tackle several SAC limitations using:

### 🧩 1. **TD3-Style Delayed Updates**
Delays policy updates to let the **critic stabilize**, similar to a **closed-loop control system**.  
✅ Improves stability  
✅ Reduces catastrophic updates

---

### 📉 2. **KL Divergence-Inspired Loss**
Borrowing from **TRPO**, this introduces controlled policy shifts using a **KL penalty**, preventing over-aggressive updates.  
Keeps your learning stable while encouraging meaningful exploration. 🧭

---

### 🧠 3. **Multi-Head Attention + Sequential Integration**
Brings the power of **transformers** and **sequence modeling** to RL:
- Recognizes long-term dependencies 🔁  
- Adapts better in sparse & tricky environments 🕸️  
- Rarely used in RL but *hugely effective* here!

---

### 🛠️ 4. **Natural Reward Shaping**

High-dimensional environments = **sparse and noisy rewards**.  
This model uses **minimal and meaningful reward shaping**, preserving learning integrity while gently guiding the agent.

```python
def shaped_reward(reward, state, max_reward):
    is_falling = (state[2] < 0.8 or abs(state[3]) > 0.4 or abs(state[4]) > 0.4)
    if is_falling:
        reward += 3.0 * (1 - abs(state[3])) + 0.5 * (0.8 - state[2])
    else:
        reward += 0.05 + 1.5 * state[1] + 0.3 * state[2]**2
        reward += 1.5 * state[0] * (1 + state[2])
        reward -= 0.1 * np.square(state[3:6]).sum()
        reward -= 0.1 * (state[3]**2 + state[4]**2)
    return reward / 10.0
🎯 Outcome: The rewards feel organic, not forced or inflated.
```

🧬 5. Dedicated Feature Extractor
In high-dim spaces, simply increasing neurons can cause instability and gradient issues.
Instead, a dedicated feature extractor isolates key patterns early, making the model:

More robust

Easier to debug

Less prone to overfitting 🎯

🌟 Summary of Benefits
✅ Stable training in tough environments
✅ Efficient learning in sparse reward settings
✅ Reduced training time and lower compute costs compared to models like Dreamer V2

📸 Glimpse of Results
![image](https://github.com/user-attachments/assets/2b585872-c49c-4214-b831-d5f745075233)


📄 Full Documentation
👉 

Feel free to ⭐️ this repo or fork it if you’re diving into advanced continuous control or curious about applying attention in RL.[Enhancing Decision.docx](https://github.com/user-attachments/files/20591550/Enhancing.Decision.docx)
