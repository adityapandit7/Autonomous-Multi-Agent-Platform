const express = require('express');
const multer = require('multer');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

// Simple code refactoring function
function refactorCode(code) {
    let lines = code.split('\n');
    let refactored = [];
    
    for (let line of lines) {
        // Add proper indentation
        if (line.includes('{') || line.includes('}')) {
            refactored.push(line.trim());
        } 
        // Format function declarations
        else if (line.includes('function')) {
            refactored.push(line.replace(/\s+/g, ' ').trim());
        }
        // Add semicolons if missing
        else if (line.trim() && !line.trim().endsWith(';') && 
                 !line.trim().endsWith('{') && !line.trim().endsWith('}')) {
            refactored.push(line.trim() + ';');
        }
        else {
            refactored.push(line);
        }
    }
    
    return refactored.join('\n');
}

// Simple documentation function
function documentCode(code) {
    let lines = code.split('\n');
    let documented = [];
    
    for (let line of lines) {
        // Add comments before functions
        if (line.includes('function') && !line.trim().startsWith('//')) {
            let funcName = line.match(/function\s+(\w+)/);
            if (funcName) {
                documented.push(`/**`);
                documented.push(` * Function: ${funcName[1]}`);
                documented.push(` * Description: [Add description here]`);
                documented.push(` * Parameters: [Add parameters here]`);
                documented.push(` * Returns: [Add return value here]`);
                documented.push(` */`);
            }
        }
        documented.push(line);
    }
    
    return documented.join('\n');
}

app.post('/api/process-file', upload.single('file'), (req, res) => {
    const file = req.file;
    const option = req.body.option;
    
    if (!file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const originalContent = file.buffer.toString('utf-8');
    let processedContent = originalContent;
    let refactoredContent = null;
    let documentedContent = null;
    let changes = 0;
    
    // Apply processing based on option
    switch(option) {
        case 'refactor':
            processedContent = refactorCode(originalContent);
            changes = processedContent.length - originalContent.length;
            break;
        case 'document':
            processedContent = documentCode(originalContent);
            changes = (processedContent.match(/\/\*\*/g) || []).length;
            break;
        case 'both':
            refactoredContent = refactorCode(originalContent);
            documentedContent = documentCode(refactoredContent);
            processedContent = documentedContent; // For backward compatibility
            changes = documentedContent.length - originalContent.length;
            break;
        default:
            return res.status(400).json({ error: 'Invalid option' });
    }
    
    // Send back ALL the data
    res.json({
        originalContent: originalContent,  // NOW INCLUDED!
        processedContent: processedContent,
        refactoredContent: refactoredContent,  // For 'both' option
        documentedContent: documentedContent,  // For 'both' option
        fileName: `processed-${file.originalname}`,
        summary: `File successfully processed with option: ${option}`,
        option: option,  // Send back which option was used
        stats: {
            originalLines: originalContent.split('\n').length,
            processedLines: processedContent.split('\n').length,
            changes: changes
        }
    });
});

app.listen(3001, () => {
    console.log('Server running on port 3001');
});