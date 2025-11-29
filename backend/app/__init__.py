"""
Flask application factory
"""
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.config import Config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()


def create_app(config_class=Config):
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app)  # Enable CORS for React frontend
    
    # Register blueprints
    from app.routes.locations import bp as locations_bp
    from app.routes.statistics import bp as statistics_bp
    from app.routes.counts import bp as counts_bp
    from app.routes.comparison import bp as comparison_bp
    from app.routes.export import bp as export_bp
    
    app.register_blueprint(locations_bp, url_prefix='/api/locations')
    app.register_blueprint(statistics_bp, url_prefix='/api/statistics')
    app.register_blueprint(counts_bp, url_prefix='/api/counts')
    app.register_blueprint(comparison_bp, url_prefix='/api/comparison')
    app.register_blueprint(export_bp, url_prefix='/api/export')
    
    # Health check endpoint
    @app.route('/api/health')
    def health():
        return {'status': 'healthy', 'service': 'pedestrian-count-api'}, 200
    
    return app

