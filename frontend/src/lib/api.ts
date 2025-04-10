import { API_CONFIG } from './config';

interface FetchDataParams {
  n?: string | number;
}

export async function fetchData(params: FetchDataParams = {}) {
  const { n = API_CONFIG.DEFAULT_RECORDS } = params;
  
  const url = `${API_CONFIG.BASE_URL}/data?n=${n}`;
  
  console.log('Fetching data from:', url);
  
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'token': API_CONFIG.TOKEN,
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', response.status, errorText);
      throw new Error(`Failed to fetch data: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('API fetch error:', error);
    throw error;
  }
}

export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  
  const url = `${API_CONFIG.BASE_URL}/upload`;
  console.log('Uploading to:', url);
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'token': API_CONFIG.TOKEN,
      },
      mode: 'cors',
      body: formData
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', response.status, errorText);
      throw new Error(`Failed to upload file: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('API upload error:', error);
    throw error;
  }
} 