const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS
app.use(cors());

// Serve Static Files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Fallback to index.html for SPA-like feel (optional, but good practice)
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Frontend UI Server running at http://localhost:${PORT}`);
    console.log(`Make sure the Python Backend is running on port 8000`);
});
