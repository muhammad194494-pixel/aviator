#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, sys

UPLOAD_DIR = "/home/muhammad194494/aviator"
UPLOAD_FILE = os.path.join(UPLOAD_DIR, "aviator_data_live.json")

class UploadHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400); self.end_headers(); return
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            if not isinstance(data, list):
                self.send_response(400); self.end_headers()
                self.wfile.write(b"Invalid JSON - must be a list")
                return
            with open(UPLOAD_FILE, 'w') as f:
                json.dump(data, f)
            self.send_response(200); self.end_headers()
            self.wfile.write(f"Data diterima: {len(data)} entries".encode())
            print(f"[+] Data live diperbarui: {len(data)} entries")
        except Exception as e:
            self.send_response(500); self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        pass  # biar gak spam

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7003
    server = HTTPServer(('0.0.0.0', port), UploadHandler)
    print(f"Server upload siap di port {port}")
    server.serve_forever()
