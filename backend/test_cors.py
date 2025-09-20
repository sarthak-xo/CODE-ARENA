from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
import re
from typing import Dict, Any, Optional
import logging
import html

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/evaluate": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/generate-questions": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# OpenRouter API configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "meta-llama/llama-4-maverick:free"

class CodeEvaluator:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-maverick:free"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:5001",
            "X-Title": "Code Evaluation Tool"
        }

    def evaluate_code(self, code: str, language: str, question: Optional[str] = None) -> Dict[str, Any]:
        prompt = self._build_evaluation_prompt(code, language, question)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a code evaluation expert. Analyze code for correctness, efficiency, and best practices."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.1
        }
        
        logger.debug(f"Sending request to {self.api_url}")
        logger.debug(f"Headers: {self.headers}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            logger.debug(f"Response status code: {response.status_code}")
            
            response_json = None
            try:
                response_json = response.json()
                logger.debug(f"Response JSON: {json.dumps(response_json, indent=2)}")
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            
            if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                evaluation = response_json["choices"][0]["message"]["content"]
                
                plain_text_evaluation = self._convert_html_to_text(evaluation)
                
                grade = self._extract_grade(evaluation)
                if grade is None:
                    grade = self._calculate_grade(evaluation)
                
                return {
                    "success": True,
                    "evaluation": plain_text_evaluation,
                    "grade": grade,
                    "result": "success",
                    "review": plain_text_evaluation
                }
            else:
                error_msg = "No evaluation returned from API"
                if response_json:
                    error_msg += f": {json.dumps(response_json)}"
                logger.error(error_msg)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "grade": 0,
                    "result": "error",
                    "review": "Failed to evaluate code. Please try again later."
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "grade": 0,
                "result": "error",
                "review": "API request failed. Please check your connection or try again later."
            }
            
        except Exception as e:
            error_msg = f"Unexpected error during evaluation: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "grade": 0,
                "result": "error",
                "review": "An unexpected error occurred. Please try again later."
            }

    def generate_questions(self, language: str, topic: str, difficulty: str, num_questions: int) -> Dict[str, Any]:
        prompt = f"""
Generate {num_questions} programming questions for {language} on the topic of {topic} with {difficulty} difficulty level.
Each question should be concise (max 200 characters) and suitable for a coding assignment.
Return the questions as a JSON array of strings, e.g., ["question1", "question2", ...].
Ensure questions are clear, specific, and test relevant concepts for the given topic and difficulty.
"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a programming question generator. Create clear and concise coding questions in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                content = response_json["choices"][0]["message"]["content"]
                logger.debug(f"Raw AI response: {content}")
                
                # Try to parse the content as JSON
                try:
                    # Handle cases where response is wrapped in code fences or other formatting
                    content = content.strip()
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()
                    elif content.startswith("```") and content.endswith("```"):
                        content = content[3:-3].strip()
                    
                    questions = json.loads(content)
                    if not isinstance(questions, list):
                        raise ValueError("Response is not a list")
                    
                    # Validate and clean questions
                    valid_questions = [
                        q for q in questions 
                        if isinstance(q, str) and len(q.strip()) <= 200 and len(q.strip()) > 0
                    ]
                    
                    if not valid_questions:
                        raise ValueError("No valid questions generated")
                    
                    return {
                        "success": True,
                        "questions": valid_questions
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse questions JSON: {str(e)}")
                    # Fallback: attempt to extract questions from plain text
                    fallback_questions = self._extract_questions_from_text(content)
                    if fallback_questions:
                        return {
                            "success": True,
                            "questions": fallback_questions
                        }
                    return {
                        "success": False,
                        "error": f"Invalid JSON response: {str(e)}",
                        "questions": []
                    }
                except ValueError as e:
                    logger.error(f"Invalid questions format: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "questions": []
                    }
            else:
                logger.error("No questions returned from API")
                return {
                    "success": False,
                    "error": "No questions returned from API",
                    "questions": []
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "questions": []
            }
            
        except Exception as e:
            logger.error(f"Unexpected error during question generation: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "questions": []
            }

    def _extract_questions_from_text(self, text: str) -> list:
        """Attempt to extract questions from plain text as a fallback."""
        lines = text.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            # Look for lines that seem like questions (e.g., starting with a number or bullet)
            if re.match(r'^\d+\.\s|^\*\s|^-\s', line):
                question = re.sub(r'^\d+\.\s|^\*\s|^-\s', '', line).strip()
                if len(question) <= 200 and len(question) > 0:
                    questions.append(question)
        return questions[:10]  # Limit to max 10 questions

    def _build_evaluation_prompt(self, code: str, language: str, question: Optional[str] = None) -> str:
        if not language or language.strip() == "":
            language = self._detect_language(code)
            
        if question:
            prompt = f"""
Please evaluate the following {language} code that is supposed to solve this problem:

PROBLEM:
{question}

CODE:
```{language}
{code}
```

Provide a detailed evaluation covering:
1. Correctness: Does the code correctly solve the problem?
2. Efficiency: Is the code efficient? Are there any performance concerns?
3. Code quality: Is the code well-structured, readable, and maintainable?
4. Best practices: Does the code follow language-specific best practices?
5. Suggestions for improvement

Please conclude with a numerical grade out of 10 for the overall code quality.
End your review with a clear "Final Grade: X/10" where X is your numerical assessment.

Format your response as plain text. DO NOT use HTML tags like <h2>, <p>, <ul>, etc.
Instead, use Markdown formatting like ## for headings, * for bullet points.
"""
        else:
            prompt = f"""
Please evaluate the following {language} code:

```{language}
{code}
```

Provide a detailed evaluation covering:
1. Functionality: What does this code do?
2. Efficiency: Is the code efficient? Are there any performance concerns?
3. Code quality: Is the code well-structured, readable, and maintainable?
4. Best practices: Does the code follow language-specific best practices?
5. Suggestions for improvement

Please conclude with a numerical grade out of 10 for the overall code quality.
End your review with a clear "Final Grade: X/10" where X is your numerical assessment.

Format your response as plain text. DO NOT use HTML tags like <h2>, <p>, <ul>, etc.
Instead, use Markdown formatting like ## for headings, * for bullet points.
"""
        return prompt

    def _detect_language(self, code: str) -> str:
        code = code.lower()
        
        if "def " in code and ":" in code:
            return "python"
        elif "function" in code and ("{" in code or "=>" in code):
            return "javascript"
        elif "<html" in code or "</div>" in code:
            return "html"
        elif "public class" in code or "private void" in code:
            return "java"
        elif "#include" in code and ("int main" in code or "void main" in code):
            return "c++"
        elif "package main" in code or "func " in code and "{" in code:
            return "go"
        else:
            return "code"

    def _convert_html_to_text(self, text: str) -> str:
        if re.search(r'<[^>]+>', text):
            text = re.sub(r'<h1>(.*?)</h1>', r'# \1\n', text)
            text = re.sub(r'<h2>(.*?)</h2>', r'## \1\n', text)
            text = re.sub(r'<h3>(.*?)</h3>', r'### \1\n', text)
            text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', text)
            text = re.sub(r'<ul>(.*?)</ul>', r'\1\n', text, flags=re.DOTALL)
            text = re.sub(r'<li>(.*?)</li>', r'* \1\n', text)
            text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
            text = re.sub(r'<pre><code>(.*?)</code></pre>', r'```\n\1\n```', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            text = html.unescape(text)
            text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    def _extract_grade(self, evaluation: str) -> Optional[float]:
        patterns = [
            r'Final Grade:\s*(\d+(?:\.\d+)?)/10',
            r'Grade:\s*(\d+(?:\.\d+)?)/10',
            r'Overall Grade:\s*(\d+(?:\.\d+)?)/10',
            r'Score:\s*(\d+(?:\.\d+)?)/10',
            r'Rating:\s*(\d+(?:\.\d+)?)/10',
            r'grade of (\d+(?:\.\d+)?)/10',
            r'grade: (\d+(?:\.\d+)?)/10',
            r'(\d+(?:\.\d+)?)/10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, evaluation, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _calculate_grade(self, evaluation: str) -> float:
        grade = 7.0
        
        positive_indicators = [
            "excellent", "great", "well structured", "efficient", "clean",
            "maintainable", "good practice", "best practice", "well organized"
        ]
        
        negative_indicators = [
            "error", "bug", "issue", "inefficient", "confusing", "poor",
            "bad practice", "fix", "problem", "security vulnerability",
            "missing", "redundant", "unnecessary"
        ]
        
        positive_count = sum(evaluation.lower().count(indicator) for indicator in positive_indicators)
        negative_count = sum(evaluation.lower().count(indicator) for indicator in negative_indicators)
        
        adjustment = min(2.5, positive_count * 0.2) - min(4, negative_count * 0.25)
        grade = max(1, min(10, grade + adjustment))
        
        return round(grade, 1)

evaluator = CodeEvaluator(OPENROUTER_API_KEY, OPENROUTER_MODEL)

@app.route('/evaluate', methods=['OPTIONS'])
@app.route('/generate-questions', methods=['OPTIONS'])
def options():
    return '', 200

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    logger.info(f"Received evaluation request: {json.dumps(data, indent=2)}")
    
    question = data.get('question', '')
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    if not code:
        logger.warning("Evaluation request received with no code")
        return jsonify({
            "success": False,
            "error": "No code provided",
            "grade": 0,
            "result": "error",
            "review": "No code provided for evaluation"
        }), 400
    
    result = evaluator.evaluate_code(code, language, question)
    logger.info(f"Evaluation completed with success={result.get('success', False)}, grade={result.get('grade', 'N/A')}")
    
    return jsonify(result)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    logger.info(f"Received question generation request: {json.dumps(data, indent=2)}")
    
    language = data.get('language', '')
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'Medium')
    num_questions = data.get('numQuestions', 1)
    
    if not language or not topic or not num_questions:
        logger.warning("Incomplete question generation parameters")
        return jsonify({
            "success": False,
            "error": "Missing required parameters",
            "questions": []
        }), 400
    
    result = evaluator.generate_questions(language, topic, difficulty, num_questions)
    logger.info(f"Question generation completed with success={result.get('success', False)}, questions_count={len(result.get('questions', []))}")
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"})

if __name__ == '__main__':
    logger.info("Starting Code Evaluation and Question Generation API server on port 5001")
    app.run(debug=True, port=5001)