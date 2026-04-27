"""
AI Debate Loop — Dual API Proxy Server v2
CORS 헤더 명시적 처리 버전
"""

import os
import requests
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# ── CORS 헤더 — 모든 응답에 명시적으로 추가 ──────────────
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"]       = "86400"
    return response

app.after_request(add_cors)

# ── OPTIONS Preflight 일괄 처리 ───────────────────────────
@app.route("/", methods=["OPTIONS"])
@app.route("/gemini", methods=["OPTIONS"])
@app.route("/claude", methods=["OPTIONS"])
def options_handler():
    return make_response("", 204)

# ── 환경변수 설정 ──────────────────────────────────────
# Claude API 키 (선택: 서버에 고정하거나 요청 시 전달)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PORT = int(os.environ.get("PORT", 5001))


# ══════════════════════════════════════════════════════
# 헬스체크
# ══════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "endpoints": ["/gemini", "/claude"],
        "claude_key_set": bool(ANTHROPIC_API_KEY)
    })


# ══════════════════════════════════════════════════════
# Gemini 프록시
# POST /gemini
# Body: { "api_key": "AIza...", "query": "...", "model": "gemini-2.5-pro",
#         "file_b64": "...(optional)", "file_type": "image/jpeg(optional)" }
# ══════════════════════════════════════════════════════
@app.route("/gemini", methods=["POST"])
def gemini_proxy():
    try:
        body     = request.get_json(force=True)
        api_key  = body.get("api_key", "").strip()
        query    = body.get("query", "")
        model    = body.get("model", "gemini-2.5-pro")
        file_b64 = body.get("file_b64")
        file_type = body.get("file_type", "image/jpeg")

        if not api_key:
            return jsonify({"error": "api_key 누락"}), 400
        if not query:
            return jsonify({"error": "query 누락"}), 400

        # ── 메시지 구성 ──
        parts = []
        if file_b64:
            parts.append({
                "inline_data": {
                    "mime_type": file_type,
                    "data": file_b64
                }
            })
        parts.append({"text": query})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.7
            }
        }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{model}:generateContent?key={api_key}"
        )

        resp = requests.post(url, json=payload, timeout=120)
        data = resp.json()

        if resp.status_code != 200:
            err = data.get("error", {}).get("message", str(data))
            return jsonify({"error": f"Gemini API 오류: {err}"}), resp.status_code

        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )
        if not text:
            return jsonify({"error": "Gemini 응답 텍스트 없음", "raw": data}), 500

        return jsonify({"text": text})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Gemini API 타임아웃 (120s)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# Claude 프록시
# POST /claude
# Body: { "api_key": "sk-ant-...(optional)", "prompt": "...",
#         "model": "claude-sonnet-4-6(optional)",
#         "file_b64": "...(optional)", "file_type": "image/jpeg(optional)" }
# ══════════════════════════════════════════════════════
@app.route("/claude", methods=["POST"])
def claude_proxy():
    try:
        body      = request.get_json(force=True)
        # 요청에 키가 있으면 우선 사용, 없으면 환경변수
        api_key   = body.get("api_key", "").strip() or ANTHROPIC_API_KEY
        prompt    = body.get("prompt", "")
        model     = body.get("model", "claude-sonnet-4-6")
        file_b64  = body.get("file_b64")
        file_type = body.get("file_type", "image/jpeg")
        max_tokens = body.get("max_tokens", 2000)

        if not api_key:
            return jsonify({
                "error": "Claude API 키 없음. "
                         "서버의 ANTHROPIC_API_KEY 환경변수를 설정하거나 "
                         "요청 body에 api_key를 포함하세요."
            }), 400
        if not prompt:
            return jsonify({"error": "prompt 누락"}), 400

        # ── 메시지 구성 ──
        if file_b64:
            if file_type == "application/pdf":
                content = [
                    {"type": "document", "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": file_b64
                    }},
                    {"type": "text", "text": prompt}
                ]
            else:
                content = [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": file_type,
                        "data": file_b64
                    }},
                    {"type": "text", "text": prompt}
                ]
        else:
            content = prompt

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]
        }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            timeout=120
        )
        data = resp.json()

        if resp.status_code != 200:
            err = data.get("error", {}).get("message", str(data))
            return jsonify({"error": f"Claude API 오류: {err}"}), resp.status_code

        text = data.get("content", [{}])[0].get("text", "")
        if not text:
            return jsonify({"error": "Claude 응답 텍스트 없음", "raw": data}), 500

        return jsonify({"text": text})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Claude API 타임아웃 (120s)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════╗
║   AI Debate Loop Proxy v1.0              ║
║   http://0.0.0.0:{PORT}                    ║
║   Endpoints: /gemini  /claude            ║
╚══════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=PORT, debug=False)
