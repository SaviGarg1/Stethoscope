from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Welcome to MedTalk</h1>
    <p>This is my first healthcare website.</p>
    """

if __name__ == "__main__":
    app.run(debug=True)