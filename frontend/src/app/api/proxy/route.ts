import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

// Handle OPTIONS preflight requests
export async function OPTIONS() {
  return NextResponse.json({}, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, token, predictions',
    }
  });
}

// Handle all requests through this proxy
export async function POST(request: NextRequest) {
  return handleRequest(request);
}

export async function GET(request: NextRequest) {
  return handleRequest(request);
}

async function handleRequest(request: NextRequest) {
  try {
    // Get the URL parameters
    const { searchParams } = new URL(request.url);
    const endpoint = searchParams.get('endpoint');
    
    if (!endpoint) {
      return NextResponse.json({ error: 'Missing endpoint parameter' }, { status: 400 });
    }

    // Get backend URL from environment variable
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://api.ai-3.net';
    
    // Reconstruct URL with any other query params
    const queryParams = new URLSearchParams();
    searchParams.forEach((value, key) => {
      if (key !== 'endpoint') {
        queryParams.append(key, value);
      }
    });
    
    const queryString = queryParams.toString() ? `?${queryParams.toString()}` : '';
    const targetUrl = `${backendUrl}/${endpoint}${queryString}`;
    
    console.log(`Proxying request to: ${targetUrl}`);
    
    // Forward the request with its headers and body
    const method = request.method;
    const headers = new Headers();
    
    // Copy all headers from original request
    request.headers.forEach((value, key) => {
      // Skip headers that might cause CORS issues or that Fetch will handle
      const lower = key.toLowerCase();
      if (['host', 'origin', 'referer', 'content-type', 'content-length'].includes(lower)) {
        return;
      }
      headers.append(key, value);
    });
    
    const requestInit: RequestInit = {
      method,
      headers,
    };
    
    // Include body for POST/PUT requests
    if (['POST', 'PUT'].includes(method)) {
      const contentType = request.headers.get('content-type') || '';
      
      if (contentType.includes('multipart/form-data')) {
        // Handle form data
        try {
          const formData = await request.formData();
          console.log(`Received form data with keys: ${Array.from(formData.keys()).join(', ')}`);
          
          // Create a new FormData object for the backend request
          const backendFormData = new FormData();
          
          // Process each form field
          for (const [key, value] of formData.entries()) {
            if (value instanceof File) {
              console.log(`Processing file: ${key} (${value.name}, ${value.size} bytes, ${value.type})`);
              
              try {
                // Read the file as an ArrayBuffer
                const buffer = await value.arrayBuffer();
                console.log(`Read file buffer of ${buffer.byteLength} bytes`);
                
                // Create a new file object
                const file = new File([buffer], value.name, { 
                  type: value.type || 'application/octet-stream'
                });
                
                // Add to the new FormData
                backendFormData.append(key, file);
                console.log(`Added file to backendFormData: ${key} (${file.size} bytes)`);
              } catch (fileError) {
                console.error(`Error processing file ${value.name}:`, fileError);
                throw new Error(`Failed to process file: ${value.name}`);
              }
            } else {
              // For non-file fields, simply append to the new FormData
              backendFormData.append(key, value);
              console.log(`Added field to backendFormData: ${key} = ${value}`);
            }
          }
          
          requestInit.body = backendFormData;
          console.log('FormData prepared for backend request');
        } catch (formError) {
          console.error('Error processing form data:', formError);
          throw new Error('Failed to process form data');
        }
      } else if (contentType.includes('application/json')) {
        requestInit.body = JSON.stringify(await request.json());
      } else {
        requestInit.body = await request.text();
      }
    }
    
    console.log(`Proxy request to ${targetUrl}:`, {
      method,
      contentType: request.headers.get('content-type'),
      hasBody: !!requestInit.body,
    });
    
    // Sanity check - log the actual request we're about to make
    console.log('Proxy →', targetUrl, { 
      method, 
      headers: Object.fromEntries([...headers.entries()]),
      bodyType: requestInit.body ? (requestInit.body instanceof FormData ? 'FormData' : typeof requestInit.body) : 'none'
    });
    
    // Make the request to the backend
    console.log(`Making ${method} request to ${targetUrl}`);
    
    let response;
    try {
      response = await fetch(targetUrl, requestInit);
      console.log(`Backend response status: ${response.status} ${response.statusText}`);
    } catch (fetchError) {
      console.error(`Error connecting to ${targetUrl}:`, fetchError);
      return NextResponse.json(
        { error: `Failed to connect to backend server: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}` },
        { status: 502 }
      );
    }
    
    // Get the response data
    let responseData;
    try {
      responseData = await response.text();
      console.log(`Backend response length: ${responseData.length} characters`);
      if (responseData.length < 1000) {
        console.log(`Backend response content: ${responseData}`);
      } else {
        console.log(`Backend response content (truncated): ${responseData.substring(0, 500)}...`);
      }
    } catch (responseError) {
      console.error('Error reading response:', responseError);
      return NextResponse.json(
        { error: 'Failed to read response from backend' },
        { status: 500 }
      );
    }
    
    // Create a new response with CORS headers
    const responseInit = {
      status: response.status,
      statusText: response.statusText,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
      }
    };
    
    return new NextResponse(responseData, responseInit);
  } catch (error) {
    console.error('API proxy error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 