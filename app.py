from flask import Flask, render_template, request
import arden_to_automata as arden
import konieczny_alg as konieczny

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("idk.html", space_mode="on", processed_input="")

@app.route("/process", methods=["POST"])
def process():
    mode = request.form["mode"]
    raw_input = request.form["user_input"]
    user_input = request.form.get("processed_input", raw_input)
    processed_input = user_input
    equation = request.form.get("equation")  # ← USER INPUT
    space_mode = request.form.get("space_mode", "off")

    error = None
    svg = None
    right_svg = None
    left_svg = None
    equation_result = None

    if not user_input.strip():
        error = "Error: Input cannot be empty."
        return render_template(
            "idk.html",
            user_input=user_input,
            svg=svg,
            right_svg=right_svg,
            left_svg=left_svg,
            error=error,
            space_mode=space_mode
        )


    try:
        if space_mode == "on":
            user_input = konieczny.add_spacing(user_input)
        if mode == "arden":
            equations = arden.parse_equations(user_input)
            automaton = arden.arden_to_automata(equations)
            min_dfa = konieczny.automaton_to_pyformlang_min_dfa(automaton)


        elif mode == "regex":
            min_dfa = konieczny.regex_to_pyformlang_min_dfa(user_input)


        # --- 2. Compute semigroup + SVG ---
        svg = konieczny.visualize_syntactic_monoid(min_dfa)

        # --- 3. Compute Cayley graphs ---
        right_svg = konieczny.right_cayley_graph_svg(min_dfa)
        left_svg = konieczny.left_cayley_graph_svg(min_dfa)

        # --- 4. Solve equation if provided ---
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
        "idk.html",
        user_input=raw_input,
        processed_input=processed_input,
        svg=svg,
        right_svg=right_svg,
        left_svg=left_svg,
        error=error,
        equation=equation,
        equation_result=equation_result,
        space_mode=space_mode
    )



if __name__ == "__main__":
    app.run(debug=True)
