from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import threading
import math
import json

app = Flask(__name__)
CORS(app)

NUM_ROUNDS = 3
shared_opponent_moves = []
lock = threading.Lock()

strategies = {
    "Favor Rock": {"R": 0.6, "P": 0.2, "S": 0.2},
    "Cycle": {"R": 1.0, "P": 0.0, "S": 0.0},
    "Random": {"R": 0.33, "P": 0.33, "S": 0.34},
}

transition_matrix = {
    "Favor Rock": {"Favor Rock": 0.8, "Cycle": 0.1, "Random": 0.1},
    "Cycle": {"Cycle": 0.8, "Favor Rock": 0.1, "Random": 0.1},
    "Random": {"Random": 0.8, "Favor Rock": 0.1, "Cycle": 0.1},
}

strategy_list = list(strategies.keys())
initial_state = random.choices(strategy_list, weights=[0.33, 0.34, 0.33])[0]
state_sequence = [initial_state]

cycle_order = ["R", "P", "S"]
# cycle_index = 0

def sample_next_state(prev_state):
    trans_probs = transition_matrix[prev_state]
    return random.choices(list(trans_probs.keys()), weights=trans_probs.values())[0]

def sample_emission(state):
    global cycle_index
    if state == "Cycle":
        move = cycle_order[cycle_index % 3]
        cycle_index += 1
        return move
    probs = strategies[state]
    return random.choices(list(probs.keys()), weights=probs.values())[0]

with lock:
    curr_state = state_sequence[0]
    for i in range(NUM_ROUNDS):
        if curr_state == "Cycle":
            cycle_count = sum(1 for s in state_sequence if s == "Cycle")
            move = cycle_order[cycle_count % 3]
        else:
            move = sample_emission(curr_state)
        shared_opponent_moves.append(move)
        next_state = sample_next_state(curr_state)
        state_sequence.append(next_state)
        curr_state = next_state

user_data = {}

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/play", methods=["POST"])
def play():
    data = request.get_json()
    user_id = data.get("user_id")
    round_num = data.get("round")
    user_move = data.get("move")

    if not user_id or round_num is None or user_move not in ["R", "P", "S"]:
        return jsonify({"error": "invalid input"}), 400

    if user_id not in user_data:
        user_data[user_id] = {"moves": {}, "guessed_strategy": None}

    if round_num < 1 or round_num > NUM_ROUNDS:
        return jsonify({"error": "invalid round number"}), 400

    opponent_move = shared_opponent_moves[round_num - 1]
    user_data[user_id]["moves"][round_num] = {
        "user": user_move,
        "opponent": opponent_move,
        "result": outcome(user_move, opponent_move),
    }

    return jsonify({"opponent_move": opponent_move})

def outcome(user, opponent):
    if user == opponent:
        return "Draw"
    if (user == "R" and opponent == "S") or \
       (user == "P" and opponent == "R") or \
       (user == "S" and opponent == "P"):
        return "Win"
    return "Lose"

@app.route("/guess", methods=["POST"])
def guess():
    data = request.get_json()
    user_id = data.get("user_id")
    guess = data.get("guess")

    if guess not in strategy_list:
        return jsonify({"error": "Invalid strategy guess"}), 400
    if user_id not in user_data:
        return jsonify({"error": "User not found"}), 404

    user_data[user_id]["guessed_strategy"] = guess
    correct_strategy = state_sequence[0]

    return jsonify({
        "correct": guess == correct_strategy,
        "true_strategy": correct_strategy
    })

@app.route("/results", methods=["GET"])
def results():
    return jsonify({"users": user_data, "true_sequence": state_sequence, "moves": shared_opponent_moves})

@app.route("/hmm_analysis")
def hmm_analysis():
    observations = shared_opponent_moves
    states = strategy_list
    start_prob = {s: 1/len(states) for s in states}
    trans_prob = transition_matrix
    emit_prob = strategies

    V = [{}]
    path = {}

    for s in states:
        V[0][s] = math.log(start_prob[s] + 1e-8) + math.log(emit_prob[s][observations[0]] + 1e-8)
        path[s] = [s]

    for t in range(1, len(observations)):
        V.append({})
        newpath = {}
        for s in states:
            (prob, state) = max(
                (V[t-1][s0] + math.log(trans_prob[s0][s] + 1e-8) + math.log(emit_prob[s][observations[t]] + 1e-8), s0)
                for s0 in states
            )
            V[t][s] = prob
            newpath[s] = path[state] + [s]
        path = newpath

    (prob, state) = max((V[-1][s], s) for s in states)
    best_path = path[state]

    belief = []
    belief_details = []
    current = start_prob.copy()

    for t, obs in enumerate(observations):
        new_belief = {}
        row_details = {}
        total = 0.0
        for s in states:
            terms = [(current[s0], trans_prob[s0][s], emit_prob[s][obs], current[s0] * trans_prob[s0][s] * emit_prob[s][obs]) for s0 in states]
            prob = sum(term[3] for term in terms)
            new_belief[s] = prob
            row_details[s] = terms
            total += prob
        for s in new_belief:
            new_belief[s] /= total
        belief.append(new_belief)
        belief_details.append(row_details)
        current = new_belief

    user_table = []
    guess_hist = {s: 0 for s in states}
    for user_id, data in user_data.items():
        guess = data.get("guessed_strategy")
        correct = guess == state_sequence[0] if guess else False
        if guess and guess in guess_hist: #oops this was crashing out before
            guess_hist[guess] += 1
        row = {
            "user": user_id,
            "guess": guess,
            "correct": correct,
            "moves": data.get("moves", {})
        }
        user_table.append(row)

    return render_template("hmm_analysis.html",
                           observations=observations,
                           viterbi_path=best_path,
                           belief_table=belief,
                           belief_details=belief_details,
                           states=states,
                           user_table=user_table,
                           guess_hist=guess_hist,
                           correct_strategy=state_sequence[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)    
# NOTE: LMFAOOO do not use port 5000, its reserverd for mac airplay