// index.js
require('dotenv').config();                     // Load environment variables from .env

const express = require('express');
const multer  = require('multer');
const path    = require('path');
const { Storage } = require('@google-cloud/storage');
const mysql   = require('mysql2/promise');

const app = express();
const PORT = process.env.PORT || 3000;

// Google Cloud Storage client setup
const storage = new Storage();  
const bucketName = process.env.GCS_BUCKET_NAME;
const bucket = storage.bucket(bucketName);

// MySQL database connection pool setup
const dbConfig = {
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  waitForConnections: true,
  connectionLimit: 10,            // up to 10 concurrent connections
  queueLimit: 0
};
let pool;
(async () => {
  try {
    pool = await mysql.createPool(dbConfig);
    console.log("Connected to MySQL database.");
  } catch (err) {
    console.error("Could not connect to MySQL database:", err);
  }
})();

// Multer configuration for file uploads
const MAX_SIZE = 50 * 1024 * 1024;  // 50 MB limit

// Storage engine – store files temporarily in an 'uploads' folder
const storageEngine = multer.diskStorage({
  destination: './uploads',                           // Folder to save uploaded files (ensure it exists)
  filename: (req, file, cb) => {
    // Construct a unique filename to prevent overwriting, e.g., original name + current timestamp
    const ext = path.extname(file.originalname);
    const baseName = path.basename(file.originalname, ext);
    cb(null, `${baseName}-${Date.now()}${ext}`);
  }
});

// File type filter – only allow .xls or .xlsx files
function excelFileFilter(req, file, cb) {
  const ext = path.extname(file.originalname).toLowerCase();
  if (ext !== '.xls' && ext !== '.xlsx') {
    return cb(new Error('Only Excel files (.xls or .xlsx) are allowed'));
  }
  // (Optional) You can also check MIME type for extra validation:
  if (!file.mimetype.includes('spreadsheet') && !file.mimetype.includes('excel')) {
    return cb(new Error('File is not a valid Excel spreadsheet'));
  }
  cb(null, true);
}

// Initialize Multer with our settings
const upload = multer({
  storage: storageEngine,
  limits: { fileSize: MAX_SIZE },      // Limit file size to 50MB
  fileFilter: excelFileFilter
});

// POST /upload endpoint
app.post('/upload', (req, res) => {
    // Use the Multer middleware to handle the single file upload (field name 'file')
    upload.single('file')(req, res, async (err) => {
      // Multer error handling
      if (err) {
        if (err.message.includes('File too large') || err.code === 'LIMIT_FILE_SIZE') {
          return res.status(400).json({ error: 'File is too large. Max 50MB allowed.' });
        }
        // For other errors (invalid file type, etc.)
        return res.status(400).json({ error: err.message });
      }
      if (!req.file) {
        // No file was uploaded
        return res.status(400).json({ error: 'No file uploaded or file type not allowed.' });
      }
  
      // File has been uploaded to the server storage at req.file.path
      const localFilePath = req.file.path;
      const fileName = req.file.filename || req.file.originalname;
  
      try {
        // Upload the file from local disk to the GCS bucket
        await bucket.upload(localFilePath, { destination: fileName });
        // Optionally, delete the file from local storage after upload (to save space)
        // fs.unlinkSync(localFilePath);
  
        console.log(`Uploaded ${fileName} to GCS bucket ${bucketName}`);
        return res.status(200).json({ message: 'File uploaded successfully to cloud storage.' });
      } catch (uploadErr) {
        console.error('Error uploading to GCS:', uploadErr);
        return res.status(500).json({ error: 'Failed to upload file to cloud storage.' });
      }
    });
  });

// GET /data endpoint - fetch last N records from the database
app.get('/data', async (req, res) => {
    const N = parseInt(req.query.n || req.query.N, 10);  // accept 'n' or 'N' as query param
    if (!N || N <= 0) {
      return res.status(400).json({ error: 'Please specify a positive number N in query parameters.' });
    }
  
    try {
      // Query to get last N records, assuming 'records' table and 'id' is an auto-increment primary key
      const query = `SELECT * FROM SALES ORDER BY sale_date DESC LIMIT ?`;
      const [rows] = await pool.query(query, [N]);  // use parameter binding to prevent SQL injection
      // The query returns an array of rows
      return res.status(200).json({ data: rows });
    } catch (err) {
      console.error('Database query error:', err);
      return res.status(500).json({ error: 'Failed to retrieve data from the database.' });
    }
  });

  // Error-handling middleware (to catch any errors that slip through or unexpected exceptions)
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'An unexpected error occurred.' });
  });

  // Start the Express server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
  