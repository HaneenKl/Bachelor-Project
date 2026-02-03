from flask import Flask, render_template, request, redirect, session
from flask_session import Session
import input_to_automata as aut
import semigroup as sg
import cayley_graph as cg

app = Flask(__name__)
app.secret_key = "dev-secret"

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

def update_history():
    history = session.get("history", [])
    entry = {
        "mode": session.get("last_mode"),
        "input": session.get("last_input"),
    }
    if entry not in history:
        history.append(entry)
    session["history"] = history

def persist_input_from_request():
    mode = request.form.get("mode")
    raw_input = request.form.get("user_input")
    space_mode = request.form.get("space_mode", "off")

    if not mode or not raw_input:
        raise ValueError("No input provided")

    input_changed = (
        raw_input != session.get("last_input")
        or mode != session.get("last_mode")
        or space_mode != session.get("last_space_mode")
    )
    
    if input_changed:
        # clear ALL artifacts from previous input
        for key in [
            "eggbox_svg",
            "left_svg",
            "right_svg",
            "equation_result",
            "equation",
            "show_equations",
        ]:
            session.pop(key, None)

    session["last_input"] = raw_input
    session["last_mode"] = mode
    session["last_space_mode"] = space_mode

@app.route("/")
def home():
    return render_template(
        "index.html",
        history=session.get("history", []),
        user_input=session.get("last_input", ""),
        processed_input=session.get("processed_input", ""),
        svg=session.get("eggbox_svg"),
        left_svg=session.get("left_svg"),
        right_svg=session.get("right_svg"),
        error=session.get("error"),
        equation=session.get("equation"),
        equation_result=session.get("equation_result"),
        show_equations=session.get("show_equations", False),
        space_mode=session.get("last_space_mode", "off"),
        last_mode=session.get("last_mode")
    )

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/clear_history", methods=["POST"])
def clear_history():
    for key in [
        "history",
        "eggbox_svg",
        "left_svg",
        "right_svg",
        "equation_result",
        "equation",
        "show_equations",
        "last_input",
        "last_mode",
        "last_space_mode",
        "processed_input",
        "error",
    ]:
        session.pop(key, None)

    return redirect("/")

def build_min_dfa_from_session():
    mode = session.get("last_mode")
    user_input = session.get("last_input")
    space_mode = session.get("last_space_mode", "off")

    if not mode or not user_input:
        raise ValueError("No input available")

    if mode == "regex":
        if space_mode == "on":
            user_input = aut.add_spacing_to_regex(user_input)
        return aut.regex_to_pyformlang_min_dfa(user_input)

    if mode == "arden":
        return aut.arden_to_pyformlang_min_dfa(user_input)

    raise ValueError("Unknown mode")

@app.route("/eggbox", methods=["POST"])
def eggbox():
    try:
        persist_input_from_request()
        update_history()

        min_dfa = build_min_dfa_from_session()
        svg = sg.visualize_syntactic_monoid(min_dfa)
        session["eggbox_svg"] = svg
        session["error"] = None
    except Exception as e:
        session["error"] = f"Error computing eggbox: {e}"
    return redirect("/")

@app.route("/left_cayley", methods=["POST"])
def left_cayley():
    try:
        persist_input_from_request()
        update_history()

        min_dfa = build_min_dfa_from_session()
        left_svg = cg.left_cayley_graph_svg(min_dfa)
        session["left_svg"] = left_svg
        session["error"] = None
    except Exception as e:
        session["error"] = f"Error computing left Cayley graph: {e}"
    return redirect("/")

@app.route("/right_cayley", methods=["POST"])
def right_cayley():
    try:
        persist_input_from_request()
        update_history()

        min_dfa = build_min_dfa_from_session()
        right_svg = cg.right_cayley_graph_svg(min_dfa)
        session["right_svg"] = right_svg
        session["error"] = None
    except Exception as e:
        session["error"] = f"Error computing right Cayley graph: {e}"
    return redirect("/")

@app.route("/equations", methods=["POST"])
def equations():
    equation_input = request.form.get("equation", "")
    try:
        persist_input_from_request()
        update_history()

        min_dfa = build_min_dfa_from_session()
        if equation_input:
            elements, reps = sg.compute_syntactic_semigroup(min_dfa)
            results = sg.check_equations_batch(elements, reps, equation_input)

            output_lines = []
            for i, res in enumerate(results, start=1):
                if res["holds"]:
                    output_lines.append(f"{i}. holds")
                else:
                    line = f"{i}. fails"
                    if res.get("counterexample"):
                        ce = ", ".join(
                            f"{var} = {val}"
                            for var, val in res["counterexample"].items()
                        )
                        line += f", counterexample: {ce}"
                    if "error" in res:
                        line += f", error: {res['error']}"
                    output_lines.append(line)

            session["equation_result"] = "\n".join(output_lines)
            session["equation"] = equation_input
            session["show_equations"] = True
        session["error"] = None
    except Exception as e:
        session["error"] = f"Error checking equations: {e}"
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
