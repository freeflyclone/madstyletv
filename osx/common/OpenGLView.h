//
//  OpenGLView.m
//  osx
//
//  Created by Evan Mortimore on 8/18/16.
//
//

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import "ExampleXGL.h"

@interface OpenGLView : NSOpenGLView <NSWindowDelegate> {
    CVDisplayLinkRef mDisplayLink;
    ExampleXGL *exgl;
}

- (void) prepareOpenGL;
- (void) drawRect: (NSRect) bounds;
- (void) awakeFromNib;
- (void) reshape;
- (void) timerFired:(id)sender;
- (void) mouseDown: (NSEvent *) theEvent;
- (void) mouseUp: (NSEvent *) theEvent;
- (void) mouseDragged: (NSEvent *) theEvent;
- (void) keyDown: (NSEvent *) theEvent;
- (void) keyUp: (NSEvent *) theEvent;

@end
