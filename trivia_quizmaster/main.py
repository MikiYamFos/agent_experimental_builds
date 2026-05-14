import logfire
import questionary
from dotenv import load_dotenv
from trivia_agent import run


load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


def ask_feedback():

    result = questionary.select(
        "How was the trivia session?",
        choices=["👍 Good", "👎 Bad", "Skip"],
    ).ask()

    if result is None or result == "Skip":

        return None

    return 1 if "Good" in result else -1


def main():
    #run("Let's play 5 easy questions from Film")
    # with logfire.span("trivia_session"):
    #     run("Let's play 5 easy questions from Music")
    with logfire.span("trivia_session"):
        session_context = logfire.get_context()
        run("Let's play 5 easy questions from Science & Nature")
        feedback = ask_feedback()
        if feedback is not None:
            with logfire.attach_context(session_context):
                logfire.info("user_feedback", feedback=feedback)

if __name__ == "__main__":
    main()
