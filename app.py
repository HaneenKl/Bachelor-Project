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
    equation = request.form.get("equation")  # ← USER INPUT



    error = None
    svg = None
    equation_result = None

    if not user_input.strip():
        error = "Error: Input cannot be empty."
        return render_template(
            "index.html",
            user_input=user_input,
            error=error
        )


    try:
        if mode == "arden":
            equations = arden.parse_equations(user_input)
            automaton = arden.arden_to_automata(equations)
            min_dfa = konieczny.automaton_to_pyformlang_min_dfa(automaton)


        elif mode == "regex":
            min_dfa = konieczny.regex_to_pyformlang_min_dfa(user_input)


        # --- 2. Compute semigroup + SVG ---
        svg = konieczny.visualize_syntactic_monoid(min_dfa)

        # --- 3. Solve equation if provided ---
        if equation:
            elements, reps = konieczny.compute_syntactic_semigroup(min_dfa)

            result = konieczny.check_equation_sat(elements, reps, equation)

            if result["holds"]:
                equation_result = "Equation holds."
            else:
                lines = [
                    f"{var} = {val}"
                    for var, val in result["counterexample"].items()
                ]
                equation_result = "Equation fails:\n" + ", ".join(lines)




    except Exception as e:
        error = f"Error processing input: {str(e)}"

    return render_template(
        "index.html",
        user_input=user_input,
        svg=svg,
        error=error,
        equation=equation,
        equation_result=equation_result,
    )



if __name__ == "__main__":
    app.run(debug=True)
