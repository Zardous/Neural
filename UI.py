'''
This is a library I made to simplify UX in other parts of this project. 

If run directly, it can be used to display the weights of inputs of any neuron in any layer of the trained model. 
Often, interesting patterns are revealed. The PyGame application this runs is not meant to be a stand alone product, it is just an extra.
'''

import pygame as pg
import numpy as np
from Activators import *

if __name__=='__main__': # Initialise the application
    pg.init()
    # Screen dimensions
    windowSizePercentage = 0.8
    displayInfo = pg.display.Info()
    windowWidth = int(displayInfo.current_w * windowSizePercentage)
    windowHeight = int(displayInfo.current_h * windowSizePercentage)
    pg.display.set_caption('GUI module')
    window = pg.display.set_mode((windowWidth, windowHeight), pg.RESIZABLE)

class txtInput:
    def __init__(self, window, pixelCoords: tuple, title='', numericalOnly = False):
        self.window = window
        self.pixelCoords = pixelCoords
        self.font = pg.font.SysFont('Verdana', 20)
        self.titleFont = pg.font.SysFont('Verdana', 14)
        self.text = ''
        self.placeholder = 'Type here...'
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_interval = 500
        self.title = title
        self.numericalOnly = numericalOnly
        self.Dimensions = tuple(w+padding*2 for w in self.font.render(self.placeholder, True, highlightColour).get_size())

    def draw(self):
        displayText = self.text if self.text else self.placeholder
        textSurface = self.font.render(displayText, True, TextColour if self.text else faintTextColour)
        titleSurface = self.titleFont.render(self.title, True, highlightColour if self.active else borderColour)
        textWidth, textHeight = textSurface.get_size()
        width = max(textWidth + 2*padding, titleSurface.get_size()[0]+2*borderRadius)
        height = textHeight + 2*padding
        self.Dimensions = (width, height)

        pg.draw.rect(
            self.window, 
            boxColour, 
            pg.Rect(self.pixelCoords[0] - highlightThickness,
                    self.pixelCoords[1] - highlightThickness,
                    self.Dimensions[0] + 2 * highlightThickness,
                    self.Dimensions[1] + 2 * highlightThickness),
                    border_radius=borderRadius)
        
        if highlightThickness > 0.0:
            pg.draw.rect(
                self.window,
                highlightColour if self.active else borderColour,
                pg.Rect(self.pixelCoords[0] - highlightThickness,
                        self.pixelCoords[1] - highlightThickness,
                        self.Dimensions[0] + 2 * highlightThickness,
                        self.Dimensions[1] + 2 * highlightThickness),
                        width=highlightThickness,
                        border_radius=borderRadius
                )
            
            #mask for text
            if self.title:
                pg.draw.rect(
                    self.window,
                    boxColour,
                    pg.Rect(self.pixelCoords[0] + padding-3, 
                            self.pixelCoords[1] - highlightThickness, 
                            titleSurface.get_size()[0]+6, 
                            highlightThickness)
                    )

        self.window.blit(textSurface, (self.pixelCoords[0] + padding, self.pixelCoords[1] + padding))

        self.window.blit(titleSurface, (self.pixelCoords[0]+ padding, self.pixelCoords[1] - titleSurface.get_size()[1]/2 - highlightThickness/2))

        if pg.time.get_ticks() - self.cursor_timer >= self.cursor_interval:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = pg.time.get_ticks()

        if self.active and self.cursor_visible:
            # Calculate cursor_x based on cursorPos
            if self.text:
                cursor_x = self.pixelCoords[0] + padding + self.font.size(self.text[:getattr(self, "cursorPos", len(self.text))])[0]
            else:
                cursor_x = self.pixelCoords[0] + padding
            cursor_y = self.pixelCoords[1] + padding
            cursor_height = textHeight
            pg.draw.line(self.window, (180, 180, 180), (cursor_x, cursor_y), (cursor_x, cursor_y + cursor_height))

    def handleEvent(self, event):
        if not hasattr(self, "cursorPos"):
            self.cursorPos = len(self.text)
        if not hasattr(self, "output"):
            self.output = ""

        if event.type == pg.MOUSEBUTTONDOWN:
            if self.active:
                self.active = pg.Rect(self.pixelCoords, self.Dimensions).collidepoint(event.pos)
                if not self.active:
                    self.output = self.text
                    if self.numericalOnly and self.text:
                        self.output = float(self.output)
            else:
                self.active = pg.Rect(self.pixelCoords, self.Dimensions).collidepoint(event.pos)
            self.cursor_visible = self.active
            if self.active:
                # Set cursor position based on mouse click (approximate)
                relativeX = event.pos[0] - (self.pixelCoords[0] + padding)
                pos = 0
                for i in range(len(self.text) + 1):
                    width = self.font.size(self.text[:i])[0]
                    if relativeX < width:
                        pos = i
                        break
                    pos = i
                self.cursorPos = pos

        elif event.type == pg.KEYDOWN and self.active:
            if event.key == pg.K_BACKSPACE:
                if self.cursorPos > 0:
                    self.text = self.text[:self.cursorPos-1] + self.text[self.cursorPos:]
                    self.cursorPos -= 1
            elif event.key == pg.K_DELETE:
                if self.cursorPos < len(self.text):
                    self.text = self.text[:self.cursorPos] + self.text[self.cursorPos+1:]
            elif event.key == pg.K_LEFT:
                if self.cursorPos > 0:
                    self.cursorPos -= 1
            elif event.key == pg.K_RIGHT:
                if self.cursorPos < len(self.text):
                    self.cursorPos += 1
            elif event.key == pg.K_HOME:
                self.cursorPos = 0
            elif event.key == pg.K_END:
                self.cursorPos = len(self.text)
            else:
                try:
                    if self.numericalOnly:
                        if ord('0') <= ord(event.unicode) <= ord('9') or ord(event.unicode)==ord('.'):
                            self.text = self.text[:self.cursorPos] + event.unicode + self.text[self.cursorPos:]
                            self.cursorPos += 1
                    elif 32 <= ord(event.unicode) <= 126:
                        self.text = self.text[:self.cursorPos] + event.unicode + self.text[self.cursorPos:]
                        self.cursorPos += 1
                except TypeError:
                    print(f'Filtered key: {pg.key.name(event.key)}')
                    pass
            if event.key == pg.K_RETURN:
                self.active = False
                self.output = self.text
                if self.numericalOnly and self.text:
                    self.output = float(self.output)
            if event.key == pg.K_ESCAPE:
                self.text = str(self.output)
                self.cursorPos = len(self.text)
                self.active = False

class Slider:
    def __init__(self, window, pixelCoords: tuple, width: int = 200, minVal: float = 0.0, maxVal: float = 1.0, initial: float = 0.5):
        self.window = window
        self.pixelCoords = pixelCoords
        self.width = width
        self.height = max(2*borderRadius, padding)
        self.minVal = minVal
        self.maxVal = maxVal
        self.value = initial
        self.dragging = False
        self.handleRadius = self.height*2/3

    def draw(self):
        x, y = self.pixelCoords

        # Background
        pg.draw.rect(
            self.window, 
            boxColour,
            pg.Rect(x - highlightThickness, y - highlightThickness, self.width + 2*highlightThickness, self.height + 2*highlightThickness),
            border_radius=borderRadius
        )

        # Highlight
        if highlightThickness>0:
            pg.draw.rect(
                self.window,
                highlightColour if self.dragging else borderColour,
                pg.Rect(x - highlightThickness, y - highlightThickness, self.width + 2*highlightThickness, self.height + 2*highlightThickness),
                width=highlightThickness,
                border_radius=borderRadius
            )

        # Filled portion
        fillWidth = round((self.value - self.minVal) / (self.maxVal - self.minVal) * self.width)
        pg.draw.rect(
            self.window,
            highlightColour,
            pg.Rect(x - highlightThickness, y - highlightThickness, fillWidth + highlightThickness, self.height + 2*highlightThickness),
            border_radius=borderRadius
        )
        if fillWidth > borderRadius:
            pg.draw.rect(
                self.window,
                highlightColour,
                pg.Rect(x + borderRadius, y - highlightThickness, fillWidth - borderRadius, self.height + 2*highlightThickness),
            )

        # Handle
        handleX = x + fillWidth
        handleY = y + self.height / 2
        pg.draw.circle(self.window, highlightColour, (handleX, handleY), self.handleRadius)
        if not self.dragging:
            pg.draw.circle(self.window, boxColour, (handleX, handleY), self.height/2)

    def handleEvent(self, event):
        x, y = self.pixelCoords
        handleX = x + round((self.value - self.minVal) / (self.maxVal - self.minVal) * self.width)
        handleY = y + self.height / 2

        if event.type == pg.MOUSEBUTTONDOWN:
            if pg.Vector2(event.pos).distance_to((handleX, handleY)) <= self.handleRadius :
                self.dragging = True

        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pg.MOUSEMOTION and self.dragging:
            relativeX = event.pos[0] - x
            relativeX = max(0, min(self.width, relativeX))
            self.value = self.minVal + (relativeX / self.width) * (self.maxVal - self.minVal)

class Checkbox:
    def __init__(self, window, pixelCoords: tuple):
        self.window = window
        self.pixelCoords = pixelCoords
        self.state = False
        self.size: int = 25

    def draw(self):
        pg.draw.rect(self.window, 
            highlightColour if self.state else boxColour, 
            pg.Rect(self.pixelCoords[0] - highlightThickness,
                    self.pixelCoords[1] - highlightThickness,
                    self.size + 2 * highlightThickness,
                    self.size + 2 * highlightThickness),
                    border_radius=borderRadius)
        
        if highlightThickness>0:
            pg.draw.rect(
                self.window,
                highlightColour if self.state else borderColour,
                pg.Rect(self.pixelCoords[0] - highlightThickness,
                        self.pixelCoords[1] - highlightThickness,
                        self.size + 2 * highlightThickness,
                        self.size + 2 * highlightThickness),
                        width=highlightThickness,
                        border_radius=borderRadius)
        
        if self.state:
            # Draw a rounded check mark with proper line caps
            thickness = self.size/5

            # Coordinates for the check mark (relative to the checkbox)
            start = pg.Vector2(self.pixelCoords[0] + int(self.size * 0.22), self.pixelCoords[1] + int(self.size * 0.55))
            mid = pg.Vector2(self.pixelCoords[0] + int(self.size * 0.42), self.pixelCoords[1] + int(self.size * 0.75))
            end = pg.Vector2(self.pixelCoords[0] + int(self.size * 0.78), self.pixelCoords[1] + int(self.size * 0.28))

            def draw_line_with_caps(surface, color, p1, p2, thickness):
                direction = (p2 - p1).normalize()
                perpendicular = pg.Vector2(-direction.y, direction.x)
                offset = perpendicular * (thickness / 2)

                points = [
                    p1 + offset,
                    p2 + offset,
                    p2 - offset,
                    p1 - offset
                ]
                pg.draw.polygon(surface, color, points)
                pg.draw.circle(surface, color, p1, thickness // 2)
                pg.draw.circle(surface, color, p2, thickness // 2)

            draw_line_with_caps(self.window, boxColour, start, mid, thickness)
            draw_line_with_caps(self.window, boxColour, mid, end, thickness)



    def handleEvent(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if pg.Rect(self.pixelCoords, (self.size, self.size)).collidepoint(event.pos):
                self.state = not self.state

class plotArray:
    def __init__(self, window, pixelCoords: tuple, NParray):
        self.window = window
        self.pixelCoords = pixelCoords

        arr = NParray

        # Convert grayscale to RGB if needed
        if arr.ndim == 1:
            arr.shape += (1,)
        if arr.ndim == 2:
            red = np.clip(np.where(arr < 0, -arr, 0), 0, 255)
            green = np.zeros_like(arr)
            blue = np.clip(np.where(arr >= 0, arr, 0), 0, 255)

            arr = np.stack((red, green, blue), axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] != 3:
            raise ValueError("Only grayscale or RGB arrays are supported.")

        self.NParray = arr
        self.arrayShape = arr.shape[1], arr.shape[0]  # width, height for pygame

    def draw(self, pixelsPerEntry: float = 1.0):
        self.pixelsPerEntry = pixelsPerEntry

        arraySurf = pg.Surface(self.arrayShape)
        pg.surfarray.blit_array(arraySurf, np.transpose(self.NParray, (1, 0, 2)))  # to (width, height, 3)
        arraySurf = pg.transform.scale_by(arraySurf, self.pixelsPerEntry)

        self.window.blit(arraySurf, self.pixelCoords)

class Header:
    def __init__(self, window, pixelCoords: tuple, header):
        self.window = window
        self.pixelCoords = pixelCoords
        self.font = pg.font.SysFont('Verdana', 20)
        self.header = header
        self.Dimensions = tuple(w+padding*2 for w in self.font.render(self.header, True, highlightColour).get_size())

    def draw(self):
        textSurface = self.font.render(self.header, True, highlightColour)
        textWidth, textHeight = textSurface.get_size()
        width = textWidth + 2*padding
        height = textHeight + 2*padding
        self.Dimensions = (width, height)

        pg.draw.rect(
            self.window, 
            boxColour, 
            pg.Rect(self.pixelCoords[0] - highlightThickness,
                    self.pixelCoords[1] - highlightThickness,
                    self.Dimensions[0] + 2 * highlightThickness,
                    self.Dimensions[1] + 2 * highlightThickness),
                    border_radius=borderRadius)
        
        if highlightThickness > 0.0:
            pg.draw.rect(
                self.window,
                borderColour,
                pg.Rect(self.pixelCoords[0] - highlightThickness,
                        self.pixelCoords[1] - highlightThickness,
                        self.Dimensions[0] + 2 * highlightThickness,
                        self.Dimensions[1] + 2 * highlightThickness),
                        width=highlightThickness,
                        border_radius=borderRadius
                )

        self.window.blit(textSurface, (self.pixelCoords[0] + padding, self.pixelCoords[1] + padding))

    def handleEvent(self, event):
        pass

class Dropdown:
    def __init__(self, window, pixelCoords: tuple, default='Pick option...', options=['Choose this','Perhaps this','Or maybe this'], title = 'Choose here'):
        self.window = window
        self.pixelCoords = pixelCoords
        self.font = pg.font.SysFont('Verdana', 20)
        self.titleFont = pg.font.SysFont('Verdana', 14)
        self.title = title
        self.default = default
        self.options = [str(x) for x in options]
        self.selected = default
        self.expanded = False
        self.hovered_index = -1
        self.optionHeight = self.font.get_height() + 2 * padding
        self.width = 200

    def draw(self):
        titleSurface = self.titleFont.render(self.title, True, highlightColour if self.expanded else borderColour)

        if self.expanded:
            rect = pg.Rect(*self.pixelCoords, self.width, self.optionHeight*len(self.options))
            pg.draw.rect(self.window, boxColour, rect, border_radius=borderRadius)
            if highlightThickness > 0.0:
                pg.draw.rect(
                    self.window,
                    highlightColour if self.expanded else borderColour,
                    pg.Rect(self.pixelCoords[0] - highlightThickness,
                            self.pixelCoords[1] - highlightThickness,
                            self.width + 2 * highlightThickness,
                            self.optionHeight*len(self.options) + 2 * highlightThickness),
                            width=highlightThickness,
                            border_radius=borderRadius
                    )
            for i, option in enumerate(self.options):
                pos = (self.pixelCoords[0], self.pixelCoords[1] + i * self.optionHeight)
                if i == self.hovered_index:
                    rect = pg.Rect(*pos, self.width, self.optionHeight)
                    pg.draw.rect(self.window, boxSelectedColour, rect, border_radius=borderRadius)
                self._draw_option(option, pos, TextColour)

        else:
            rect = pg.Rect(*self.pixelCoords, self.width, self.optionHeight)
            pg.draw.rect(self.window, boxColour, rect, border_radius=borderRadius)
            if highlightThickness > 0.0:
                pg.draw.rect(
                    self.window,
                    highlightColour if self.expanded else borderColour,
                    pg.Rect(self.pixelCoords[0] - highlightThickness,
                            self.pixelCoords[1] - highlightThickness,
                            self.width + 2 * highlightThickness,
                            self.optionHeight + 2 * highlightThickness),
                            width=highlightThickness,
                            border_radius=borderRadius
                    )
            self._draw_option(self.selected, self.pixelCoords, TextColour if self.selected != self.default else faintTextColour)
        
        if self.title:
            pg.draw.rect(
                self.window,
                boxColour,
                pg.Rect(self.pixelCoords[0] + padding-3, 
                        self.pixelCoords[1] - highlightThickness, 
                        titleSurface.get_size()[0]+6, 
                        highlightThickness)
                )
            self.window.blit(titleSurface, (self.pixelCoords[0]+ padding, self.pixelCoords[1] - titleSurface.get_size()[1]/2 - highlightThickness/2))

    def _draw_option(self, text, pos, colour):
        textSurface = self.font.render(text, True, colour)
        self.window.blit(textSurface, (pos[0] + padding, pos[1] + padding))

    def handleEvent(self, event):
        mouse_pos = pg.mouse.get_pos()
        x, y = self.pixelCoords
        base_rect = pg.Rect(x, y, self.width, self.optionHeight)

        if event.type == pg.MOUSEBUTTONDOWN:
            if base_rect.collidepoint(mouse_pos) and not self.expanded:
                self.expanded = not self.expanded
            elif self.expanded:
                for i, option in enumerate(self.options):
                    opt_rect = pg.Rect(x, y + i * self.optionHeight, self.width, self.optionHeight)
                    if opt_rect.collidepoint(mouse_pos):
                        self.selected = option
                        self.expanded = False
                        break
                else:
                    self.expanded = False  # Clicked outside

        elif event.type == pg.MOUSEMOTION and self.expanded:
            self.hovered_index = -1
            for i, option in enumerate(self.options):
                opt_rect = pg.Rect(x, y + i * self.optionHeight, self.width, self.optionHeight)
                if opt_rect.collidepoint(mouse_pos):
                    self.hovered_index = i
                    break

class Button:
    def __init__(self, window, pixelCoords: tuple, text, reset=0):
        self.window = window
        self.pixelCoords = pixelCoords
        self.state = False
        self.font = pg.font.SysFont('Verdana', 20)
        self.text = text
        self.reset = reset
        self.pressTime = -1
        self.hovering = False
        self.Dimensions = tuple(w+padding*2 for w in self.font.render(self.text, True, highlightColour).get_size())

    def draw(self):
        textSurface = self.font.render(self.text, True, highlightColour if self.state else (TextColour if self.hovering else faintTextColour))
        textWidth, textHeight = textSurface.get_size()
        width = textWidth + 2*padding
        height = textHeight + 2*padding
        self.Dimensions = (width, height)

        if pg.mouse.get_pressed()[0] and pg.Rect(self.pixelCoords, self.Dimensions).collidepoint(pg.mouse.get_pos()):
            self.pressTime = pg.time.get_ticks()
        if self.reset>0 and self.pressTime != -1:
            if pg.time.get_ticks() - self.pressTime >= self.reset:
                self.state = False

        pg.draw.rect(
            self.window, 
            boxSelectedColour if self.state else (boxSelectedColour if self.hovering else boxColour), 
            pg.Rect(self.pixelCoords[0] - highlightThickness,
                    self.pixelCoords[1] - highlightThickness,
                    self.Dimensions[0] + 2 * highlightThickness,
                    self.Dimensions[1] + 2 * highlightThickness),
                    border_radius=borderRadius)
        
        if highlightThickness > 0.0:
            pg.draw.rect(
                self.window,
                highlightColour if self.state else borderColour,
                pg.Rect(self.pixelCoords[0] - highlightThickness,
                        self.pixelCoords[1] - highlightThickness,
                        self.Dimensions[0] + 2 * highlightThickness,
                        self.Dimensions[1] + 2 * highlightThickness),
                        width=highlightThickness,
                        border_radius=borderRadius
                )
            
        self.window.blit(textSurface, (self.pixelCoords[0] + padding, self.pixelCoords[1] + padding))

    def handleEvent(self, event):
        if event.type == pg.MOUSEBUTTONDOWN: 
            if pg.Rect(self.pixelCoords, self.Dimensions).collidepoint(event.pos):
                self.state = not self.state
        elif event.type == pg.MOUSEMOTION:
                if pg.Rect(self.pixelCoords, self.Dimensions).collidepoint(pg.mouse.get_pos()):
                    self.hovering = True
                else:
                    self.hovering = False

# Colours
bgColour = 30, 30, 30
TextColour = 150, 150, 150
faintTextColour = 50, 50, 50
borderColour = 75, 75, 75
boxColour = 25, 25, 25
boxSelectedColour = 35, 35, 35
highlightColour = 60, 100, 170

# Sizes
highlightThickness = 0
borderRadius = 8
padding = max(borderRadius,6)

if __name__=='__main__': # Do not run when importing:
    # Define elements on the screen
    box = txtInput(window, (50, 50))
    box.text = 'Trained model'
    box.output = box.text
    box.title =  'Text input box'
    terminal = txtInput(window, (50, 125))
    terminal.text = 'matrix = ReLU(matrix)'
    terminal.output = terminal.text
    terminal.title = 'Terminal'
    check = Checkbox(window, (50,200))
    check.state = True
    slider = Slider(window, (50, 275), 150, 1, 20, 20)
    dropdown = Dropdown(window, (50, 350), 'Choose layer', [0,1], 'Choose layer')
    button = Button(window, (50,425), 'Press', 50)
    neuronSlider = Slider(window, (50, 500), 150, 1, 127, 0)

    # Text cursor
    pg.key.set_repeat(500,33)


    clock = pg.time.Clock()
    running = True
    while running: # Main loop
        window.fill(bgColour)
        
        # Check for updates for UI
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            box.handleEvent(event)
            check.handleEvent(event)
            terminal.handleEvent(event)
            slider.handleEvent(event)
            dropdown.handleEvent(event)
            button.handleEvent(event)
            neuronSlider.handleEvent(event)


        if not box.output: # Set the default output of the box element to 0 to load neuron 0
            box.output= 0
        if dropdown.selected == dropdown.default:
            dropdown.selected='0' # Set the default output of the dropdown element to 0 to load layer 0
        try:
            matrix = np.load(rf"{box.output}\weights[{str(dropdown.selected)}].npz")['weights'][int(neuronSlider.value)]
        except Exception as e:
            print(e)
        try:
            matrix = matrix.reshape(40,40,3) # Reshape the 2D matrix to RGB
        except:
            pass        
        
        # Draw UI elements to the screen (except for the array, that needs some extra calculation)
        box.draw()
        check.draw()
        terminal.draw()
        slider.draw()
        dropdown.draw()
        button.draw()
        neuronSlider.draw()

        # Try to execute the code thats in the terminal element
        try: 
            if terminal.output:
                exec(terminal.output)
        # If that didnt work, print the error and let the user type again
        except Exception as e: 
            print(f"Error: {e}")
            terminal.output = '' # Make sure the terminal wont run the faulty command again until the user presses enter or clicks away
            terminal.active = True

        # Update the RGB matrix with the scale defined by the slider, then draw it
        scale = slider.value
        array = plotArray(window, (400, 50), matrix*255)
        if check.state:
            array.draw(scale)

        pg.display.flip()
        clock.tick(70)

    pg.quit()
