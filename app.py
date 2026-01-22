from flask import Flask, render_template, request
import input_to_automata as aut
import semi_group as sg
import cayley_graph as cg

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
    equation = request.form.get("equation")
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
        if mode == "arden":
            min_dfa = aut.arden_to_pyformlang_min_dfa(user_input)

        elif mode == "regex":
            if space_mode == "on":
                user_input = aut.add_spacing_to_regex(user_input)
            min_dfa = aut.regex_to_pyformlang_min_dfa(user_input)

        svg = sg.visualize_syntactic_monoid(min_dfa)

        right_svg = cg.right_cayley_graph_svg(min_dfa)
        left_svg = cg.left_cayley_graph_svg(min_dfa)

        if equation:
            elements, reps = sg.compute_syntactic_semigroup(min_dfa)

            result = sg.check_equation_sat(elements, reps, equation)

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
