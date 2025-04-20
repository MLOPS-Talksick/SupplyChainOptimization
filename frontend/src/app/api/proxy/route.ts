import { NextRequest, NextResponse } from 'next/server';

// Configure this route to use the Edge Runtime
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
      return NextResponse.json({ error: 'No endpoint specified' }, { status: 400 });
    }

    // Get backend URL from environment variable
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://34.8.2.47';
    
    // Reconstruct URL with any other query params
    const queryParams = new URLSearchParams();
    searchParams.forEach((value, key) => {
      if (key !== 'endpoint') {
        queryParams.append(key, value);
      }
    });
    
    const queryString = queryParams.toString() ? `?${queryParams.toString()}` : '';
    const targetUrl = `${backendUrl}/${endpoint}${queryString}`;
    
    // Forward the request with its headers and body
    const method = request.method;
    const headers = new Headers();
    
    // Copy all headers from original request
    request.headers.forEach((value, key) => {
      // Skip headers that might cause CORS issues
      if (!['host', 'origin', 'referer'].includes(key.toLowerCase())) {
        headers.append(key, value);
      }
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
        requestInit.body = await request.formData();
      } else if (contentType.includes('application/json')) {
        requestInit.body = JSON.stringify(await request.json());
      } else {
        requestInit.body = await request.text();
      }
    }
    
    // Make the request to the backend
    const response = await fetch(targetUrl, requestInit);
    
    // Get the response data
    const data = await response.clone().text();
    
    // Create a new response with CORS headers
    const responseInit = {
      status: response.status,
      statusText: response.statusText,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': response.headers.get('Content-Type') || 'application/json',
      }
    };
    
    return new NextResponse(data, responseInit);
  } catch (error) {
    console.error('API proxy error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 