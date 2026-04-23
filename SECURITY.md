# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email us at: **[Create a GitHub Security Advisory](https://github.com/monarchdev14/CARDIOVASCULAR-RISK-ASSESSMENT/security/advisories/new)**
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue
5. Allow reasonable time for a fix before public disclosure

## Security Considerations

### Patient Data Privacy
- This application processes patient health data
- No patient data is stored permanently on the server
- All predictions are computed in-memory and discarded after response
- The homomorphic encryption module (`breast_cancer_HE.py`) provides a framework for privacy-preserving inference

### Model Security
- The ML model is trained from publicly available UCI Heart Disease data
- No private or proprietary patient data is used in training
- Model weights are recomputed on each server startup

### Deployment Recommendations
- Always run behind HTTPS in production
- Use environment variables for sensitive configuration
- Enable Flask's production mode (disable debug mode)
- Implement rate limiting for the `/api/predict` endpoint
- Add authentication for clinical deployments

## Data Handling

This application:
- ✅ Processes data in-memory only
- ✅ Does not persist patient data to disk
- ✅ Does not transmit data to external services
- ✅ Uses publicly available training data
- ❌ Should NOT be used as sole clinical decision tool

---

Thank you for helping keep this project and its users secure.
