//
//  OpenGLView.m
//  osx
//
//  Created by Evan Mortimore on 8/18/16.
//
//

#import "OpenGLView.h"
#import <OpenGL/gl3.h>

@implementation OpenGLView

- (void)prepareOpenGL{
    GLint swapInt = 1;
    [[self openGLContext] setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
}

- (void)awakeFromNib {
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize, 24,
        NSOpenGLPFAOpenGLProfile,
        NSOpenGLProfileVersion3_2Core,
        NSOpenGLPFASampleBuffers,1,
        NSOpenGLPFASamples,4,
        NSOpenGLPFASampleAlpha,1,
        0
    };
    
    NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    if (!pf)
        NSLog(@"No OpenGL pixel format");
    
    NSOpenGLContext *context = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
    NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    currentWorkingDir = std::string([cwd UTF8String]);
    
    [self setPixelFormat:pf];
    [self setOpenGLContext:context];
    [self activateContext];
    
    try {
        NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        currentWorkingDir = std::string([cwd UTF8String]);
        
        exgl = new ExampleXGL();
    }
    catch (XGLException& e) {
        NSString *aStr = [NSString stringWithUTF8String: e.what()];
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:aStr];
        [alert runModal];
        abort();
    }
    
    NSTimer *renderTimer = [NSTimer timerWithTimeInterval:0.01 target:self selector:@selector(timerFired:) userInfo:nil repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:renderTimer forMode:NSDefaultRunLoopMode];
    [[NSRunLoop currentRunLoop] addTimer:renderTimer forMode:NSEventTrackingRunLoopMode];
    
    [self.window makeFirstResponder:self];
    
    NSLog(@"Set context complete.");
}
- (void) timerFired: (id) sender {
    [self setNeedsDisplay:YES];
}
- (void) mouseDown: (NSEvent *) theEvent {
}
- (void) mouseUp: (NSEvent *) theEvent {
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    if (buttons==0) {
        exgl->MouseEvent(0,0,0);
    }
}

- (void) mouseDragged: (NSEvent *) theEvent {
    NSPoint loc = [theEvent locationInWindow];
    NSUInteger buttons = [NSEvent pressedMouseButtons];
    
    exgl->MouseEvent((int)loc.x, -(int)loc.y, (int)buttons);
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
    [self activateContext];
    
    exgl->Display();
    
    [self flushContext];
}

- (void)reshape {
    if (exgl == nil) {
        return;
    }
    
    [self activateContext];
    NSRect bounds = [self bounds];
    GLsizei w = NSWidth(bounds);
    GLsizei h = NSHeight(bounds);
    exgl->Reshape(w,h);
}

- (void)activateContext {
    [[self openGLContext] makeCurrentContext];
}

- (void)flushContext {
    [[self openGLContext] flushBuffer];
}
@end
