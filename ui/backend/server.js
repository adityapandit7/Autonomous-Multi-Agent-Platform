/**
 * server.js — Express backend for CodeRefine
 *
 * Integration strategy: APPROACH 1 (Node spawns Python)
 * ───────────────────────────────────────────────────────
 * Node receives the uploaded file, writes it to a temp path,
 * spawns `python pipeline_api.py --file <path> --mode <mode>`,
 * parses the JSON stdout, cleans up, then returns normalised JSON.
 *
 * This keeps all model logic in Python and avoids the complexity of
 * running a separate Flask service.  If startup latency becomes a
 * problem, switch to the Flask proxy approach (see pipeline_api.py
 * _start_http_server) and replace _callPython with a fetch() call.
 *
 * ─── Request ────────────────────────────────────────────────────────
 * POST /api/process-file
 * Content-Type: multipart/form-data
 *
 * Fields:
 *   file    (required)  The uploaded source file
 *   option  (required)  "refactor" | "document" | "both"
 *
 * ─── Success Response ───────────────────────────────────────────────
 * HTTP 200  application/json
 * {
 *   "success": true,
 *   "mode": "refactor" | "document" | "both",
 *   "filename": string,
 *   "originalContent": string,
 *   "processedContent": string,
 *   "refactoredContent": string | null,
 *   "documentedContent": string | null,
 *   "option": string,            // mirror of mode — backward-compat with React UI
 *   "stats": {                   // backward-compat with ResultDisplay component
 *     "originalLines": number,
 *     "processedLines": number,
 *     "changes": number
 *   },
 *   "artifacts": {
 *     "runDir": string | null,
 *     "parsedAnalysis": object | null,
 *     "smellReport": string | null,
 *     "refactoringPlan": string | null,
 *     "prompts": { "refactor": string | null, "doc": string | null },
 *     "evaluation": {
 *       "refactor": object | null,
 *       "doc": object | null,
 *       "summary": object | null
 *     }
 *   },
 *   "error": null
 * }
 *
 * ─── Error Response ─────────────────────────────────────────────────
 * HTTP 400 | 500  application/json
 * {
 *   "success": false,
 *   "error": {
 *     "message": string,
 *     "type": "validation" | "runtime" | "pipeline" | "api",
 *     "details": string | object | null
 *   }
 * }
 */

const express  = require('express');
const multer   = require('multer');
const cors     = require('cors');
const path     = require('path');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

// ── Multer: store in memory (we write to a tmp file before Python call) ──────
const upload = multer({ storage: multer.memoryStorage() });

// ── Constants ────────────────────────────────────────────────────────────────
const VALID_OPTIONS   = new Set(['refactor', 'document', 'both']);
const PYTHON_BIN      = process.env.PYTHON_BIN || 'python';   // or 'python3'
const PIPELINE_SCRIPT = path.resolve(__dirname, '..', '..', 'pipeline_api.py');
const PYTHON_TIMEOUT  = parseInt(process.env.PYTHON_TIMEOUT_MS || '600000', 10); // 10 min

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Spawn the Python pipeline, passing source code via stdin as JSON.
 * This avoids all temp-file path / argument-parsing issues entirely.
 *
 * Python side reads:  json.loads(sys.stdin.read())
 * Stdin payload:      { "sourceCode": "...", "mode": "...", "filename": "..." }
 *
 * @param {Buffer} fileBuffer  raw file bytes
 * @param {string} mode        "refactor" | "document" | "both"
 * @param {string} filename    original filename (display only)
 * @returns {Promise<object>}
 */
function callPython(fileBuffer, mode, filename) {
  return new Promise((resolve, reject) => {
    const args        = [PIPELINE_SCRIPT, '--stdin', '--mode', mode];
    const projectRoot = path.dirname(PIPELINE_SCRIPT);
    const child = spawn(PYTHON_BIN, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: projectRoot,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },  // force UTF-8 stdout on Windows
    });

    // Write the payload to stdin then close it so Python's read() unblocks
    const payload = JSON.stringify({
      sourceCode: fileBuffer.toString('utf-8'),
      mode,
      filename,
    });
    child.stdin.write(payload);
    child.stdin.end();

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      reject(new Error(`Python process timed out after ${PYTHON_TIMEOUT}ms`));
    }, PYTHON_TIMEOUT);

    child.on('close', (code) => {
      clearTimeout(timer);

      if (code !== 0) {
        return reject(new Error(
          `Python exited with code ${code}.\n${stderr.slice(-2000)}`
        ));
      }

      // Extract JSON after the unique delimiter printed by pipeline_api.py
      const delimiter = '__PIPELINE_JSON__';
      const delimIdx  = stdout.indexOf(delimiter);
      const jsonStr   = delimIdx !== -1
        ? stdout.slice(delimIdx + delimiter.length).trim()
        : stdout.slice(stdout.lastIndexOf('\n{')).trim();

      if (!jsonStr.startsWith('{')) {
        return reject(new Error(
          `Python produced no JSON output.\nstdout: ${stdout.slice(-800)}`
        ));
      }

      try {
        const raw = JSON.parse(jsonStr);
        // Slim down — only send what the UI needs
        resolve(raw);
      } catch (parseErr) {
        reject(new Error(
          `Failed to parse Python JSON: ${parseErr.message}\n` +
          `stdout tail: ${stdout.slice(-800)}`
        ));
      }
    });

    child.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });
  });
}

/**
 * Build the `stats` object expected by the existing ResultDisplay component.
 */
function buildStats(originalContent, processedContent) {
  const origLines = (originalContent  || '').split('\n').length;
  const procLines = (processedContent || '').split('\n').length;
  return {
    originalLines: origLines,
    processedLines: procLines,
    changes: procLines - origLines,
  };
}

/**
 * Wrap any error into the standard error-response shape.
 *
 * @param {'validation'|'runtime'|'pipeline'|'api'} type
 * @param {string}      message
 * @param {string|object|null} details
 */
function errorResponse(type, message, details = null) {
  return { success: false, error: { message, type, details } };
}

// ── Route ─────────────────────────────────────────────────────────────────────

app.post('/api/process-file', upload.single('file'), async (req, res) => {
  // ── 1. Input validation ────────────────────────────────────────────────────
  const file   = req.file;
  const option = (req.body.option || '').trim().toLowerCase();

  if (!file) {
    return res.status(400).json(errorResponse('validation', 'No file uploaded'));
  }

  if (!option) {
    return res.status(400).json(errorResponse('validation', '"option" field is required'));
  }

  if (!VALID_OPTIONS.has(option)) {
    return res.status(400).json(errorResponse(
      'validation',
      `"option" must be one of: ${[...VALID_OPTIONS].join(', ')}`,
      { received: option }
    ));
  }

  // ── 2. Call Python pipeline (source code sent via stdin) ──────────────────
  let pyResult;
  try {
    pyResult = await callPython(file.buffer, option, file.originalname);
  } catch (err) {
    return res.status(500).json(errorResponse('pipeline', err.message, err.message));
  }

  // ── 4. Handle Python-side errors ───────────────────────────────────────────
  if (!pyResult.success) {
    const pyErr = pyResult.error || {};
    const status = pyErr.type === 'validation' ? 400 : 500;
    return res.status(status).json({
      success: false,
      error: {
        message: pyErr.message || 'Pipeline error',
        type:    pyErr.type    || 'pipeline',
        details: pyErr.details || null,
      },
    });
  }

  // ── 5. Normalise and return ────────────────────────────────────────────────
  const {
    originalContent,
    processedContent,
    refactoredContent,
    documentedContent,
    artifacts,
  } = pyResult;

  return res.status(200).json({
    success:           true,
    option:            option,
    originalContent:   pyResult.originalContent  || '',
    processedContent:  pyResult.processedContent || '',
    refactoredContent: pyResult.refactoredContent || null,
    documentedContent: pyResult.documentedContent || null,
  });
});

// ── Health check (useful for Docker / CI) ────────────────────────────────────
app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ─────────────────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || '3001', 10);
app.listen(PORT, () => {
  console.log(`[server] Listening on port ${PORT}`);
  console.log(`[server] Python binary : ${PYTHON_BIN}`);
  console.log(`[server] Pipeline script: ${PIPELINE_SCRIPT}`);
});