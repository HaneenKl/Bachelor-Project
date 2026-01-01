from flask import Flask, render_template, request
import arden_to_automata as arden
import konieczny_alg as konieczny

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    mode = request.form["mode"]
    user_input = request.form["user_input"]

    if not user_input.strip():
        return render_template("result.html", input=user_input, output="Error: Input cannot be empty.")

    svg = None  # for the automaton image

    try:
        if mode == "arden":
            equations = arden.parse_equations(user_input)
            automaton = arden.arden_to_automata(equations)
            min_dfa = konieczny.automaton_to_pyformlang_min_dfa(automaton)


            svg = konieczny.visualize_syntactic_monoid(min_dfa)

        elif mode == "regex":
            min_dfa = konieczny.regex_to_pyformlang_min_dfa(user_input)

            svg = konieczny.visualize_syntactic_monoid(min_dfa)


    except Exception as e:
        return render_template("result.html", input=user_input, output=f"Error processing input: {str(e)}")

    return render_template("result.html", input=user_input, svg=svg)



if __name__ == "__main__":
    app.run(debug=True)
