import { NextResponse } from 'next/server';

// Configure this route to use the Edge Runtime
export const runtime = 'edge';

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS() {
  return NextResponse.json({}, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, token, predictions',
    }
  });
}

export async function GET(request: Request) {
  try {
    // Parse URL to get query parameters
    const { searchParams } = new URL(request.url);
    const n = searchParams.get('n') || '50';
    
    // Get headers
    const token = request.headers.get('token');
    const predictions = request.headers.get('predictions') || 'False';
    
    // Validate token
    if (token !== 'backendapi1234567890') {
      return NextResponse.json(
        { error: 'Unauthorized access' },
        { status: 401 }
      );
    }

    // Get backend URL from environment variable
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://34.8.2.47';
    
    // Forward the request to the actual backend API
    const response = await fetch(`${backendUrl}/data?n=${n}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'token': 'backendapi1234567890',
        'predictions': predictions,
      },
      cache: 'no-store', // Prevent caching to avoid stale data
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || 'Failed to get data' },
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
    console.error('Data API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 