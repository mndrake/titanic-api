from flask import Flask, abort, request, jsonify
from scoring import survived_probability


application = Flask(__name__)


@application.route('/survived', methods=['POST'])
def survived():
    if not request.json:
        abort(400)
    return jsonify({'probability': survived_probability(request.json)})


if __name__ == "__main__":
    application.run(port=5000)