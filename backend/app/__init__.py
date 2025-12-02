"""
Flask application factory
"""
import logging
from datetime import datetime
from flask import Flask, request, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app.config import Config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    limiter.init_app(app)
    CORS(app)  # Enable CORS for React frontend
    
    # Request/Response logging middleware
    @app.before_request
    def log_request_info():
        g.start_time = datetime.now()
        logger.info(f"Request: {request.method} {request.path} | IP: {get_remote_address()} | Params: {dict(request.args)}")
    
    @app.after_request
    def log_response_info(response):
        duration = (datetime.now() - g.start_time).total_seconds()
        logger.info(f"Response: {request.method} {request.path} | Status: {response.status_code} | Duration: {duration:.3f}s")
        return response
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        logger.warning(f"Rate limit exceeded: {request.method} {request.path} | IP: {get_remote_address()}")
        return {'error': 'Rate limit exceeded. Please try again later.'}, 429
    
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

