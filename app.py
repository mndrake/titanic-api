from flask import Flask, abort, request, jsonify
from scoring import survived_probability


app = Flask(__name__)


@app.route('/survived', methods=['POST'])
def survived():
    if not request.json:
        abort(400)
    return jsonify({'probability': survived_probability(request.json)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
