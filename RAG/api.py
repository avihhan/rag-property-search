import os
import io
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables (needed for OpenAI/Pinecone)
load_dotenv()

# Import functions (keep API layer thin; no business logic here)
try:
    from .company_embed import (
        process_company_data_from_records,
        parse_companies_from_csv,
        get_index_details,
        clear_index,
    )
    from .company_search import (
        get_top_k_companies,
        parse_filter_params,
    )
except Exception:
    try:
        from RAG.company_embed import (
            process_company_data_from_records,
            parse_companies_from_csv,
            get_index_details,
            clear_index,
        )
        from RAG.company_search import (
            get_top_k_companies,
            parse_filter_params,
        )
    except Exception:
        from company_embed import (
            process_company_data_from_records,
            parse_companies_from_csv,
            get_index_details,
            clear_index,
        )
        from company_search import (
            get_top_k_companies,
            parse_filter_params,
        )


app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint to test if the server is running
    """
    return jsonify({
        'status': 'running',
        'endpoints': {
            'ingest': '/ingest (POST)',
            'search': '/search (GET)',
            'index_details': '/index-details (GET)',
            'clear_index': '/clear-index (POST/DELETE)'
        }
    }), 200


@app.route('/ingest', methods=['POST'])
def ingest_companies():
    """
    Ingest companies from an uploaded JSON/CSV file or JSON payload.
    - Multipart/form-data: field name 'file' with .json or .csv; optional 'index_name'
    - application/json: body with {'companies': [...]} or top-level list; optional query 'index_name'
    """
    index_name = (
        request.form.get('index_name')
        or request.args.get('index_name')
        or 'company-information-dummy'
    )

    companies = None

    # File upload path
    if 'file' in request.files:
        uploaded = request.files['file']
        filename = (uploaded.filename or '').lower()

        try:
            if filename.endswith('.csv'):
                companies = parse_companies_from_csv(uploaded)
            else:
                # Assume JSON by default
                # Some servers require reading bytes then decoding
                try:
                    companies = json.load(uploaded)
                except Exception:
                    uploaded.seek(0)
                    raw = uploaded.read()
                    companies = json.loads(raw.decode('utf-8'))
        except Exception as e:
            return jsonify({'error': f'Failed to parse uploaded file: {str(e)}'}), 400

    # JSON body path
    if companies is None:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({'error': 'No file uploaded and no JSON body provided'}), 400
        if isinstance(payload, list):
            companies = payload
        else:
            companies = payload.get('companies')
        if not companies:
            return jsonify({'error': 'No companies found in JSON body'}), 400

    result = process_company_data_from_records(companies, index_name=index_name)
    return jsonify(result), 200


@app.route('/index-details', methods=['GET'])
def index_details():
    """
    Return index statistics and sample structure.
    Query params:
      - index_name: required
      - sample_limit: optional (default 1)
    """
    index_name = request.args.get('index_name')
    if not index_name:
        return jsonify({'error': 'index_name is required'}), 400
    try:
        sample_limit = int(request.args.get('sample_limit', '1'))
    except Exception:
        sample_limit = 1

    details = get_index_details(index_name=index_name, sample_limit=sample_limit)
    return jsonify(details), 200


@app.route('/clear-index', methods=['POST', 'DELETE'])
def clear_index_route():
    """
    Clear vectors from an index, or delete the index entirely.
    Accepts JSON body or query params:
      - index_name: required
      - delete_index: optional (bool) — if true, deletes index itself
    """
    payload = request.get_json(silent=True) or {}
    index_name = payload.get('index_name') or request.args.get('index_name')
    if not index_name:
        return jsonify({'error': 'index_name is required'}), 400

    delete_index_param = payload.get('delete_index')
    if delete_index_param is None:
        delete_index_param = request.args.get('delete_index')
    if isinstance(delete_index_param, str):
        delete_index_flag = delete_index_param.lower() in ['true', '1', 'yes']
    else:
        delete_index_flag = bool(delete_index_param)

    result = clear_index(index_name=index_name, delete_index=delete_index_flag)
    return jsonify(result), 200


@app.route('/search', methods=['GET'])
def search_companies():
    """
    Search for companies with various filters and optional reasoning.
    
    Query parameters:
      - query: required (search query text)
      - top_k: optional (number of results, default 5)
      - index_name: optional (Pinecone index name, default 'company-information-dummy')
      - with_reasoning: optional (include reasoning, default false)
      - industry_list: optional (comma-separated list of industries)
      - location_list: optional (comma-separated list of locations)
      - revenue_min: optional (minimum revenue, e.g., "$100M")
      - revenue_max: optional (maximum revenue, e.g., "$500M")
      - employees_min: optional (minimum number of employees)
      - employees_max: optional (maximum number of employees)
    """
    # Required parameters
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is required'}), 400
    
    # Optional parameters with defaults
    try:
        top_k = int(request.args.get('top_k', '5'))
    except ValueError:
        top_k = 5
    
    index_name = request.args.get('index_name', 'company-information-dummy')
    
    # Parse boolean parameter
    with_reasoning_param = request.args.get('with_reasoning', 'false').lower()
    with_reasoning = with_reasoning_param in ['true', '1', 'yes']
    
    # Parse filter parameters
    industry_filter, location_filter, revenue_filter, employees_filter = parse_filter_params(
        industry_list=request.args.get('industry_list'),
        location_list=request.args.get('location_list'),
        revenue_min=request.args.get('revenue_min'),
        revenue_max=request.args.get('revenue_max'),
        employees_min=request.args.get('employees_min'),
        employees_max=request.args.get('employees_max')
    )
    
    # Perform search
    result = get_top_k_companies(
        query=query,
        top_k=top_k,
        industry_filter=industry_filter,
        location_filter=location_filter,
        revenue_filter=revenue_filter,
        employees_filter=employees_filter,
        with_reasoning=with_reasoning,
        index_name=index_name
    )
    
    return jsonify(result), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port)


