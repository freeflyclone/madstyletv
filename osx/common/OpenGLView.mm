//
//  OpenGLView.m
//  osx
//
//  Created by Evan Mortimore on 8/18/16.
//
//

#import "OpenGLView.h"
#import <Foundation/FOundation.h>
#import <OpenGL/gl3.h>
#import <Cocoa/Cocoa.h>
#import <CoreVideo/CVDisplayLink.h>

@implementation OpenGLView


static CVReturn displayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *now, const CVTimeStamp *outputTime, CVOptionFlags flagsIn, CVOptionFlags *flagsOut, void *displayLinkContext) {

    CVReturn result = [(__bridge OpenGLView*)displayLinkContext getFrameForTime:outputTime];
    return result;
}

- (void)prepareOpenGL{
    GLint swapInt = 1;
    [[self openGLContext] setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
    
    CVDisplayLinkCreateWithActiveCGDisplays(&mDisplayLink);
    
    CVDisplayLinkSetOutputCallback(mDisplayLink, &displayLinkCallback, (__bridge void *)self);
    CVDisplayLinkStart(mDisplayLink);
}

- (CVReturn)getFrameForTime:(const CVTimeStamp*)outputTime {
    [context lock];
    [self activateContext];
    
    if (exgl != nil)
        exgl->Display();
    
    [self flushContext];
    [context unlock];

    
    return kCVReturnSuccess;
}

- (void)awakeFromNib {
    xprintf("awakeFromNib()\n");
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize, 24,
        NSOpenGLPFAOpenGLProfile,
        NSOpenGLProfileVersion3_2Core,
        NSOpenGLPFAAccelerated,
        0
    };
    
    NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    if (!pf)
        NSLog(@"No OpenGL pixel format");
    
    context = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
    NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    currentWorkingDir = std::string([cwd UTF8String]);
    
    
    // "assets" are found in the root folder of the project for all OSes supported.
    // assume we're running under the debugger...
    pathToAssets = currentWorkingDir.substr(0, currentWorkingDir.rfind("/osx"));
    
    // if that didn't work out, assume we're running from PROJECT_DIR/bin
    if (pathToAssets == currentWorkingDir)
        pathToAssets = currentWorkingDir.substr(0, currentWorkingDir.rfind("/bin"));

    xprintf("pathToAssets: %s", pathToAssets.c_str());
    
    [self setPixelFormat:pf];
    [self setOpenGLContext:context];
    [self activateContext];
    
    [context lock];
    
    try {
        exgl = new ExampleXGL();
    }
    catch (XGLException& e) {
        NSString *aStr = [NSString stringWithUTF8String: e.what()];
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:aStr];
        [alert runModal];
        abort();
    }
    
    [self.window makeFirstResponder:self];
    
    [context unlock];
    
    xprintf("Set context complete.");
}

- (void) mouseDown: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    
    exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, (int)buttons);
}

- (void) mouseUp: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    if (buttons==0) {
        exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, 0);
    }
}

- (void) mouseDragged: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    
    exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, (int)buttons);
}

- (void) rightMouseUp: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    if (buttons==0) {
        exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, 0);
    }
}

- (void) rightMouseDown: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    
    exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, (int)buttons);
}

- (void) rightMouseDragged: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    
    exgl->MouseEvent((int)loc.x, exgl->height - (int)loc.y, (int)buttons);
}

- (void) keyDown: (NSEvent *) theEvent {
    NSString *chars = [theEvent characters];
    NSString *tmpChars = [chars uppercaseString];
    const char *cmd = [tmpChars UTF8String];
    
    exgl->KeyEvent(cmd[0], 0);
}

- (void) keyUp: (NSEvent *) theEvent {
    NSString *chars = [theEvent characters];
    NSString *tmpChars = [chars uppercaseString];
    const char *cmd = [tmpChars UTF8String];
    
    exgl->KeyEvent(cmd[0], 0x8000);
}

- (void)drawRect:(NSRect)bounds {
    [context lock];
    [self activateContext];
    
    if (exgl != nil)
        exgl->Display();
    
    [self flushContext];
    [context unlock];
}

- (void)reshape {
    if (exgl == nil) {
        return;
    }
    
    [context lock];
    
    [self activateContext];
    NSRect bounds = [self bounds];
    GLsizei w = NSWidth(bounds);
    GLsizei h = NSHeight(bounds);
    exgl->Reshape(w,h);
    
    [context unlock];
}

- (void)activateContext {
    [[self openGLContext] makeCurrentContext];
}

- (void)flushContext {
    [[self openGLContext] flushBuffer];
}

- (void)dealloc {
    CVDisplayLinkStop(mDisplayLink);
    CVDisplayLinkRelease(mDisplayLink);
}
@end
