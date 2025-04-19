import { NextResponse } from 'next/server';

// Handle preflight requests
export async function OPTIONS() {
  return NextResponse.json({}, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, token',
    }
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Verify request has necessary authorization
    const token = request.headers.get('token');
    if (token !== 'backendapi1234567890') {
      return NextResponse.json(
        { error: 'Unauthorized access' },
        { status: 401 }
      );
    }

    // Forward the request to the actual API
    const response = await fetch('http://34.8.2.47/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'token': 'backendapi1234567890',
      },
      body: JSON.stringify(body),
      cache: 'no-store', // Prevent caching to avoid stale data
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || 'Failed to get forecast data' },
        { status: response.status }
      );
    }

    // Return response with CORS headers
    return NextResponse.json(data, {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-store, max-age=0',
      }
    });
  } catch (error) {
    console.error('Forecast API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 