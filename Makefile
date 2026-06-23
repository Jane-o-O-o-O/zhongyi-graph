.PHONY: test-backend test-frontend test

test-backend:
	cd backend && python -m pytest -q

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend
