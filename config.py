from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Update with your database credentials
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:rRvV1234@localhost/your_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#
#host=localhost port=5432 dbname=alzheimers_db user=postgres sslmode=prefer con
nect_timeout=10
db = SQLAlchemy(app)
