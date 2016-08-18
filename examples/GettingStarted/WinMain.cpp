// -----------------------------------------------------------------
// WinMain.cpp : Defines the entry point for the application.
//
//   This file is mostly just standard Windows Application
//   boilerplate code, with additional code to initialize
//	 the XGL framework.
// -----------------------------------------------------------------

#include "stdafx.h"
#include "WinMain.h"

#define MAX_LOADSTRING 100

// Global Variables:
HWND hWnd;
HINSTANCE hInst;								// current instance
TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];			// the main window class name

// the derived (from XGL) OpenGL class for Windows native C++
ExampleXGL *exgl;

// Forward declarations of functions included in this code module:
ATOM				MyRegisterClass(HINSTANCE hInstance);
BOOL				InitInstance(HINSTANCE, int);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPTSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

 	// TODO: Place code here.
	MSG msg;
	HACCEL hAccelTable;
	HDC hdc;

	// Initialize global strings
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_GETTINGSTARTED, szWindowClass, MAX_LOADSTRING);

	MyRegisterClass(hInstance);

	// Perform application initialization:
	if (!InitInstance (hInstance, nCmdShow)) {
		return FALSE;
	}

	hdc = GetDC(hWnd);
	hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_GETTINGSTARTED));

	// Main message loop:
	while (TRUE) {
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))	{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		
		if (msg.message == WM_QUIT)
			break;

		if (exgl) {
			exgl->Display();
			SwapBuffers(hdc);
		}
	}

	return (int) msg.wParam;
}

//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//--------------------------------------
ATOM MyRegisterClass(HINSTANCE hInstance) {
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wcex.lpfnWndProc	= WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= LoadIcon(hInstance, MAKEINTRESOURCE(IDI_GETTINGSTARTED));
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground  = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wcex.lpszMenuName	= MAKEINTRESOURCE(IDC_GETTINGSTARTED);
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassEx(&wcex);
}

//------------------------------------------------------------------------------
//  FUNCTION: CreateOpenGLContext()
//
//  PURPOSE: Initializes the OpenGL 3.2 context prior to creating an XGL object.
//
//	 This is an OS-specific task, that's why it's here.
//------------------------------------------------------------------------------
void CreateOpenGLContext(HWND hWnd)
{
	HDC hdc;
	int pixelFormat;
	HGLRC hrc;

	static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,
		32,    // bit depth
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		16,    // z-buffer depth
		0, 0, 0, 0, 0, 0, 0,
	};

	hdc = GetDC(hWnd);

	// Pixel format.
	pixelFormat = ChoosePixelFormat(hdc, &pfd);
	SetPixelFormat(hdc, pixelFormat, &pfd);

	// Create the OpenGL Rendering Context.
	HGLRC tmp_hrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, tmp_hrc);

	glewExperimental = GL_TRUE;

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		MessageBox(hWnd, _T("GLEW is not initialized!"), _T("Well that didn't work out"), MB_ABORTRETRYIGNORE);
	}

	int attribs[] = {
		WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
		WGL_CONTEXT_MINOR_VERSION_ARB, 2,
		WGL_CONTEXT_FLAGS_ARB, 0,
		0
	};

	if (wglewIsSupported("WGL_ARB_create_context") == 1) {
		hrc = wglCreateContextAttribsARB(hdc, 0, attribs);
		wglMakeCurrent(NULL, NULL);
		wglDeleteContext(tmp_hrc);
		wglMakeCurrent(hdc, hrc);
	}
	else {
		// XGL is not coded for supporting less than full retained mode, so don't even try
		MessageBox(hWnd, _T("Unable to initialize an OpenGL version 2.1 context.  Unable to continue."), _T("Well that didn't work out"), MB_ABORTRETRYIGNORE);
		exit(0);
	}

	// force V-Sync enable.  can override with NVidia control panel.
	wglSwapIntervalEXT(1);
}

//----------------------------------------------------------------------
//  FUNCTION: SetGlobalWorkingDirectoryName()
//
//  PURPOSE: Registers the window class.
//
//	 required for exception handling with XGL.
//	 if assets cannot be found it may be because the assets
//	 provided by the project source code are not relative
//	 to the executable's current working directory.
//	 Setting "currentWorkingDir"  may help with diagnosing that problem
//	 using the exception handler's output message.
//
//	 This is an OS-specific task, that's why it's here.
//----------------------------------------------------------------------
void SetGlobalWorkingDirectoryName()
{
	DWORD sizeNeeded = GetCurrentDirectory(0, NULL);
	DWORD size;
	TCHAR *buff = new TCHAR[sizeNeeded];

	if ((size = GetCurrentDirectory(sizeNeeded, buff)) != sizeNeeded - 1)
		throwXGLException("GetCurrentDirectory() unexpectedly failed. " + std::to_string(size) + " vs " + std::to_string(sizeNeeded));

#ifdef UNICODE
	std::wstring wstr(buff);
	currentWorkingDir = std::string(wstr.begin(), wstr.end());
#else
	currentWorkingDir = std::string(buff);
#endif
	delete[] buff;
}

//------------------------------------------------------------------------------
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//------------------------------------------------------------------------------
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
   hInst = hInstance; // Store instance handle in our global variable

   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);
   if (!hWnd) {
      return FALSE;
   }

   {   // --- begin XGL setup ---
	   CreateOpenGLContext(hWnd);

	   // get the currentWorkingDirectory
	   SetGlobalWorkingDirectoryName();

	   // try to create a new XGL based object, and hopefully find out why
	   // if doing so failed.
	   try {
		   exgl = new ExampleXGL();
	   }
	   catch (XGLException e) {
		   MessageBoxA(NULL, e.what(), "Well that didn't work out", MB_OK);
		   exit(0);
	   }

	   // Initialize the projection matrix according to initial window size;
	   RECT rect;
	   GetClientRect(hWnd, &rect);
	   exgl->Reshape(rect.right - rect.left, rect.bottom - rect.top);

   } // --- end XGL setup ---

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//--------------------------------------------------------------------------------
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_ERASEBKGND - tell Windows we processed this, but don't actually do it
//  WM_SIZE		- reshape the OpenGL context and update the screen as user resizes
//  WM_DESTROY	- post a quit message and return
//--------------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	LRESULT retVal = FALSE;

	switch (message) {
		case WM_KEYDOWN:
		case WM_KEYUP:
		case WM_SYSKEYDOWN:
		case WM_SYSKEYUP:
			exgl->KeyEvent((int)wParam, HIWORD(lParam));
			break;

		case WM_LBUTTONDOWN:
		case WM_LBUTTONUP:
		case WM_MBUTTONDOWN:
		case WM_MBUTTONUP:
		case WM_RBUTTONDOWN:
		case WM_RBUTTONUP:
		case WM_MOUSEMOVE:
			exgl->MouseEvent(LOWORD(lParam), HIWORD(lParam), (int)wParam);
			break;

		case WM_ERASEBKGND:
			retVal = TRUE;
			break;

		case WM_SIZE:
			retVal = TRUE;
			exgl->Reshape(LOWORD(lParam), HIWORD(lParam));
			break;

		case WM_DESTROY:
			PostQuitMessage(0);
			break;

		default:
			retVal = DefWindowProc(hWnd, message, wParam, lParam);
	}
	return retVal;
}