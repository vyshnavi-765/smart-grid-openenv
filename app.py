import streamlit as st
import matplotlib.pyplot as plt
from env import SmartGridEnv
from baseline import BaselineAgent

st.set_page_config(page_title="Smart Grid AI", layout="centered")

st.title("⚡ Smart Grid AI Simulation")
st.write("AI-based electricity distribution with real-world constraints")

# Select task
task = st.selectbox("Select Difficulty", ["easy", "medium", "hard"])

if st.button("Run Simulation"):
    env = SmartGridEnv(task)
    agent = BaselineAgent()

    obs = env.reset()
    done = False

    st.subheader("📊 Simulation Output")

    step_num = 1

    while not done:
        action = agent.act(obs)

        demand = obs.demand
        allocation = action["allocations"]

        obs, reward, done, _ = env.step(
            type("A", (), action)
        )

        # Display values
        st.write(f"### Step {step_num}")
        st.write("Demand:", demand)
        st.write("Supply:", obs.supply)
        st.write("Allocation:", allocation)
        st.write("Reward:", reward.value)

        # 📈 Graph (Demand vs Allocation)
        fig, ax = plt.subplots()
        ax.bar(range(len(demand)), demand)
        ax.bar(range(len(allocation)), allocation)
        ax.set_title("Demand vs Allocation")

        st.pyplot(fig)

        st.write("---")
        step_num += 1

    st.success("Simulation Completed ✅")