//
//  OpenGLView.m
//  osx
//
//  Created by Evan Mortimore on 8/18/16.
//
//

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CVDisplayLink.h>
#import "ExampleXGL.h"

@interface OpenGLView : NSOpenGLView {
    CVDisplayLinkRef displayLink;
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
