import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

st.set_page_config(
    page_title="Q-Learning Tic-Tac-Toe",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# Q-LEARNING ENVIRONMENT
# -------------------------------

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

def get_state(board, player):
    opponent = "O" if player == "X" else "X"
    return "".join(
        "1" if cell == player else
        "-1" if cell == opponent else "0"
        for cell in board
    )

def legal_actions(board):
    return [i for i, cell in enumerate(board) if cell == ""]

def winner(board):
    for a, b, c in WINNING_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None

def update_q(q, state, action, reward, next_state=None,
             next_actions=None, alpha=0.2, gamma=0.9):
    old = q.get((state, action), 0.0)
    future = 0.0

    if next_state is not None and next_actions:
        future = max(
            q.get((next_state, a), 0.0)
            for a in next_actions
        )

    target = reward - gamma * future
    q[(state, action)] = old + alpha * (target - old)

def choose_action(q, state, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)

    values = [q.get((state, a), 0.0) for a in actions]
    best = max(values)
    choices = [
        a for a, value in zip(actions, values)
        if value == best
    ]
    return random.choice(choices)

# -------------------------------
# SELF-PLAY TRAINING
# -------------------------------

def train_agent(episodes, alpha, gamma, initial_epsilon):

    q = {}
    records = []
    epsilon = initial_epsilon

    for episode in range(episodes):

        board = [""] * 9
        player = "X"
        episode_result = "Draw"

        while True:

            state = get_state(board, player)
            actions = legal_actions(board)

            action = choose_action(q, state, actions, epsilon)
            board[action] = player

            result = winner(board)

            if result == player:
                update_q(
                    q, state, action, 1.0,
                    alpha=alpha, gamma=gamma
                )
                episode_result = "Win" if player == "X" else "Loss"
                break

            if not legal_actions(board):
                update_q(
                    q, state, action, 0.0,
                    alpha=alpha, gamma=gamma
                )
                episode_result = "Draw"
                break

            next_player = "O" if player == "X" else "X"
            next_state = get_state(board, next_player)
            next_actions = legal_actions(board)

            update_q(
                q, state, action, 0.0,
                next_state, next_actions,
                alpha, gamma
            )

            player = next_player

        records.append({
            "Episode": episode + 1,
            "Result": episode_result,
            "Epsilon": epsilon
        })

        epsilon = max(0.02, epsilon * 0.9995)

    return q, pd.DataFrame(records)
def evaluate_agent(q, games=1000):
    wins = 0
    losses = 0
    draws = 0

    for game in range(games):
        board = [""] * 9
        ai_player = "X" if game % 2 == 0 else "O"
        current_player = "X"

        while True:
            actions = legal_actions(board)

            if not actions:
                draws += 1
                break

            if current_player == ai_player:
                state = get_state(board, current_player)
                action = choose_action(q, state, actions, 0.0)
            else:
                action = random.choice(actions)

            board[action] = current_player
            result = winner(board)

            if result:
                if result == ai_player:
                    wins += 1
                else:
                    losses += 1
                break

            if not legal_actions(board):
                draws += 1
                break

            current_player = "O" if current_player == "X" else "X"

    return {
        "AI Wins": wins,
        "AI Losses": losses,
        "Draws": draws,
        "Win Rate": wins / games * 100
    }
# -------------------------------
# MODEL INITIALIZATION
# -------------------------------

if "q_table" not in st.session_state:
    with st.spinner("Training AI through self-play..."):
        q, history = train_agent(15000, 0.2, 0.9, 1.0)
        st.session_state.q_table = q
        st.session_state.history = history

if "board" not in st.session_state:
    st.session_state.board = [""] * 9

if "human" not in st.session_state:
    st.session_state.human = "X"

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "result" not in st.session_state:
    st.session_state.result = ""

if "scores" not in st.session_state:
    st.session_state.scores = {
        "Player Wins": 0,
        "AI Wins": 0,
        "Draws": 0
    }

if "games" not in st.session_state:
    st.session_state.games = 0

# -------------------------------
# GAME FUNCTIONS
# -------------------------------

def finish_game(message, score=None):
    st.session_state.result = message
    st.session_state.game_over = True

    if score:
        st.session_state.scores[score] += 1

    st.session_state.games += 1

def ai_move():

    board = st.session_state.board
    ai = "O" if st.session_state.human == "X" else "X"

    actions = legal_actions(board)

    if not actions:
        return

    state = get_state(board, ai)

    action = choose_action(
        st.session_state.q_table,
        state,
        actions,
        0
    )

    board[action] = ai

    if winner(board) == ai:
        finish_game("AI Wins! 🤖", "AI Wins")
    elif not legal_actions(board):
        finish_game("It's a Draw! 🤝", "Draws")

def make_move(position):

    if st.session_state.game_over:
        return

    board = st.session_state.board

    if board[position] != "":
        return

    board[position] = st.session_state.human

    if winner(board) == st.session_state.human:
        finish_game("Congratulations! You Won! 🎉", "Player Wins")
        return

    if not legal_actions(board):
        finish_game("It's a Draw! 🤝", "Draws")
        return

    ai_move()

def reset_game():
    st.session_state.board = [""] * 9
    st.session_state.game_over = False
    st.session_state.result = ""

    # Let AI start if the human chooses O
    if st.session_state.human == "O":
        ai_move()

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------

st.sidebar.title("🧠 Q-Learning AI")
st.sidebar.caption("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Play Against AI",
        "AI Training",
        "Performance Analytics",
        "How AI Learns",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.success("AI Model Trained")
st.sidebar.write("Algorithm: Q-Learning")
st.sidebar.write("Training: Self-play")
st.sidebar.write("Environment: Tic-Tac-Toe")

# -------------------------------
# HOME PAGE
# -------------------------------

if page == "Home":

    st.title("🧠 Q-Learning Agent")
    st.subheader("Tic-Tac-Toe Using Reinforcement Learning")

    st.write(
        "An interactive Machine Learning application "
        "where an AI learns to play Tic-Tac-Toe through "
        "trial and error using Q-Learning."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Q-Learning")
    c2.metric("Training Episodes", "15,000")
    c3.metric("Q-table States", len(st.session_state.q_table))

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Implement a Reinforcement Learning agent.
    - Train AI through self-play.
    - Learn optimal actions using Q-values.
    - Understand exploration and exploitation.
    - Evaluate AI performance.
    - Visualize the learning process.
    """)

    st.info("Navigate to Play Against AI to challenge your trained model.")

    if st.button("🎮 Start Playing", type="primary"):
        st.session_state.page_request = "Play Against AI"
        st.rerun()

# -------------------------------
# PLAY AGAINST AI
# -------------------------------

elif page == "Play Against AI":

    st.title("🎮 Play Against AI")
    st.write("Challenge your Q-Learning trained opponent.")

    selected = st.selectbox(
        "Choose your symbol",
        ["X", "O"],
        index=0 if st.session_state.human == "X" else 1
    )

    if selected != st.session_state.human:
        st.session_state.human = selected
        reset_game()
        st.rerun()

    st.write("You are:", st.session_state.human)

    ai_symbol = "O" if st.session_state.human == "X" else "X"
    st.write("AI is:", ai_symbol)

    # AI starts when player chooses O
    if (
        st.session_state.human == "O"
        and st.session_state.games == 0
        and not st.session_state.game_over
        and all(x == "" for x in st.session_state.board)
    ):
        ai_move()

    st.divider()

    st.subheader("Tic-Tac-Toe Board")

    for row in range(3):
        cols = st.columns(3)

        for col in range(3):
            index = row * 3 + col
            value = st.session_state.board[index]

            label = value if value else " "

            cols[col].button(
                label,
                key=f"cell_{row}_{col}",
                use_container_width=True,
                disabled=(
                    value != "" or st.session_state.game_over
                ),
                on_click=make_move,
                args=(index,)
            )

    if st.session_state.game_over:
        if "Player Wins" in st.session_state.result:
            st.success(st.session_state.result)
        elif "AI Wins" in st.session_state.result:
            st.error(st.session_state.result)
        else:
            st.info(st.session_state.result)

    st.button(
        "🔄 Restart Game",
        on_click=reset_game,
        type="primary"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Player Wins", st.session_state.scores["Player Wins"])
    c2.metric("AI Wins", st.session_state.scores["AI Wins"])
    c3.metric("Draws", st.session_state.scores["Draws"])

# -------------------------------
# AI TRAINING PAGE
# -------------------------------

elif page == "AI Training":

    st.title("⚙️ AI Training Centre")

    st.write("Train the agent using Reinforcement Learning.")

    episodes = st.select_slider(
        "Training Episodes",
        options=[1000, 5000, 10000, 15000, 20000, 30000],
        value=5000
    )

    alpha = st.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.2)
    gamma = st.slider("Discount Factor (Gamma)", 0.0, 1.0, 0.9)
    epsilon = st.slider("Initial Exploration Rate", 0.1, 1.0, 1.0)

    if st.button("🚀 Start AI Training", type="primary"):

        progress = st.progress(0)
        status = st.empty()

        status.write("Training your model...")

        q, history = train_agent(
            episodes, alpha, gamma, epsilon
        )

        st.session_state.q_table = q
        st.session_state.history = history

        progress.progress(100)
        status.success("Training completed successfully!")

        st.metric("Learned Q-values", len(q))
        st.metric("Episodes Completed", episodes)

        st.rerun()

    st.divider()
    st.subheader("Current Training Information")

    st.metric("Training Episodes", len(st.session_state.history))
    st.metric("Learned State-Action Pairs", len(st.session_state.q_table))

# -------------------------------
# PERFORMANCE ANALYTICS
# -------------------------------

elif page == "Performance Analytics":

    st.title("📊 Performance Analytics")

    history = st.session_state.history

    st.subheader("Training Results")

    results = history["Result"].value_counts()

    wins = results.get("Win", 0)
    draws = results.get("Draw", 0)

    st.write("Training episode outcomes (X player's perspective):")

    c1, c2, c3 = st.columns(3)

    c1.metric("X Wins", wins)
    c2.metric("O Wins", results.get("Loss", 0))
    c3.metric("Draws", draws)

    st.divider()

    st.subheader("Training Performance")

    history["Win Indicator"] = (
        history["Result"] == "Win"
    ).astype(int)

    history["Rolling Win Rate"] = (
        history["Win Indicator"].rolling(500, min_periods=1).mean()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        history["Episode"],
        history["Rolling Win Rate"]
    )
    ax.set_title("X Player Rolling Win Rate")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Win Rate")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Exploration Rate")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["Episode"], history["Epsilon"])
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate Reduction")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Q-Value Distribution")

    q_values = list(st.session_state.q_table.values())

    if q_values:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(q_values, bins=25)
        ax.set_title("Distribution of Learned Q-values")
        ax.set_xlabel("Q-value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

# -------------------------------
# HOW AI LEARNS
# -------------------------------

elif page == "How AI Learns":

    st.title("🧠 How Does the AI Learn?")

    st.subheader("What is Reinforcement Learning?")

    st.write(
        "Reinforcement Learning is a Machine Learning technique "
        "where an agent learns by interacting with an environment "
        "and receiving rewards or penalties."
    )

    st.subheader("Q-Learning")

    st.write(
        "Q-Learning is a model-free Reinforcement Learning "
        "algorithm that learns the value of taking an action "
        "in a particular state."
    )

    st.latex(
        r"Q(s,a) = Q(s,a) + \alpha "
        r"[r - \gamma \max_{a'}Q(s',a') - Q(s,a)]"
    )

    st.markdown("""
    **Parameters:**

    - Q(s,a): Current action value.
    - Alpha: Learning rate.
    - Gamma: Discount factor.
    - r: Immediate reward.
    - s: Current state.
    - s': Next state.
    - a: Selected action.
    """)

    st.subheader("Learning Process")

    st.markdown("""
    1. Observe the Tic-Tac-Toe board.
    2. Identify available moves.
    3. Select an action using epsilon-greedy.
    4. Receive a reward or penalty.
    5. Update the Q-table.
    6. Repeat until the game ends.
    """)

    st.subheader("Exploration vs Exploitation")

    st.write(
        "Exploration means trying new moves. "
        "Exploitation means selecting moves that have "
        "already learned high Q-values."
    )

# -------------------------------
# ABOUT PROJECT
# -------------------------------

elif page == "About Project":

    st.title("📚 About This Project")

    st.subheader("Project Title")

    st.write(
        "Q-Learning Agent for Tic-Tac-Toe Using Reinforcement Learning"
    )

    st.subheader("Dataset")

    st.write(
        "No traditional dataset is required. "
        "The training data is generated through self-play."
    )

    st.write(
        "Optional dataset: Tic-Tac-Toe Endgame Dataset "
        "from the UCI Machine Learning Repository."
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - NumPy
    - Pandas
    - Matplotlib
    - Reinforcement Learning
    """)

    st.subheader("Future Scope")

    st.markdown("""
    - Implement Deep Q-Learning.
    - Add multiplayer functionality.
    - Improve AI strategy.
    - Explore different reinforcement learning algorithms.
    """)

    st.success("Thank you for exploring our project!")
