import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyglet
from pyglet.window import mouse, key
import random
from math import pi, sin
import struct
from io import BytesIO
from pyglet.media import synthesis, Player, StaticSource
from pyglet.media.synthesis import Sine, Sawtooth, ADSREnvelope, LinearDecayEnvelope

class BugNN(nn.Module):
    def __init__(self):
        super(BugNN, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.dropout = nn.Dropout(p=0.05)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

    def mutate(self):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * 0.05
                param.data.add_(noise)

class Bug:
    def __init__(self, start_pos, window, is_baby=True):
        self.position = np.array(start_pos)
        self.window = window
        self.velocity = np.zeros(2)
        self.alive = True
        self.model = BugNN()
        self.is_baby = is_baby
        self.age = 0
        self.size = 0.2 if is_baby else 0.4
        self.vision_range = 15
        self.reproduction_age = 50
        parent_position = None
        reproduction_timer = 0

    def move(self, all_bugs, mouse_pos, dt):
        if not self.alive:
            return
        self.handle_mouse_collision(mouse_pos)
        if not self.alive:
            return
        self.update_position(mouse_pos, dt)
        self.handle_bug_collisions(all_bugs)
        self.age += 1
        if self.age > 15 and self.is_baby:
            self.grow_up()
        if self.age > self.reproduction_age:
            self.reproduce(all_bugs)

    def handle_mouse_collision(self, mouse_pos):
        if np.linalg.norm(mouse_pos - self.position) <= self.size:
            self.alive = False
            self.window.synthesize_mouse_death_sound()

    def handle_bug_collisions(self, all_bugs):
        for other_bug in all_bugs:
            if other_bug != self and other_bug.alive:
                if np.linalg.norm(self.position - other_bug.position) < (self.size + other_bug.size) * 0.1:
                    self.alive = False
                    other_bug.alive = True
                    self.window.synthesize_death_sound()
                    return

    def update_position(self, mouse_pos, dt):
        input_vec = np.concatenate((self.position, mouse_pos - self.position))
        inputs = torch.tensor(input_vec, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            delta = self.model(inputs).numpy().flatten()
        self.velocity = delta
        self.position += self.velocity * dt
        self.velocity *= 0.98

        if self.position[0] > 4 or self.position[0] < -4 or \
           self.position[1] > 4 or self.position[1] < -4:
            self.alive = False
            self.window.synthesize_wall_death_sound()

    def grow_up(self):
        self.is_baby = False
        self.size = 0.4

    def reproduce(self, all_bugs):
        if not self.is_baby and self.age > self.reproduction_age and random.random() < 0.5:
            close_bugs = [bug for bug in all_bugs if bug != self and not bug.is_baby and bug.alive and np.linalg.norm(bug.position - self.position) <= 0.1*self.vision_range]
            if close_bugs:
                other_bug = random.choice(close_bugs)
                parent_bug = random.choice([self, other_bug])
                new_bug_position = (self.position + other_bug.position) / 2

                # Create the new bug
                new_bug = Bug(new_bug_position, self.window, is_baby=True)
                new_bug.model.load_state_dict(parent_bug.model.state_dict())
                new_bug.model.mutate()
                new_bug.parent_position = self.position
                new_bug.reproduction_timer = 15
                self.window.add_bug(new_bug)

                # Add line information for visualization
                self.window.parent_lines.append((self.position, other_bug.position))
                self.window.parent_relationships[new_bug] = (self, other_bug)
                self.window.synthesize_ding_sound()

                # Only increase reproduction ages
                self.reproduction_age += 20
                other_bug.reproduction_age += 20
                self.window.parent_lines.append(((self.position, other_bug.position), 30))
                self.window.parent_relationships[new_bug] = (self, other_bug)

class SimulationWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_batch = pyglet.graphics.Batch()
        self.parent_lines = []
        self.ding_players = []  # Track active ding players
        self.max_ding_players = 3  # Maximum simultaneous ding players
        self.ding_sound_limit = 3  # Limit for concurrent "ding" sounds
        self.current_ding_sounds = 0  # Counter for current "ding" sounds
        self.parent_relationships = {}
        self.batch = pyglet.graphics.Batch()
        self.line_batch = pyglet.graphics.Batch()
        self.bugs = self.init_bugs(20)
        self.shapes = []
        self.vision_shapes = []  # List to hold vision circles
        for _ in range(len(self.bugs)):
            bug_shape = pyglet.shapes.Circle(300, 300, 5, color=(255, 0, 0), batch=self.batch)
            self.shapes.append(bug_shape)
            vision_shape = pyglet.shapes.Circle(300, 300, 150, color=(0, 0, 255, 5), batch=self.batch)  # Vision range circle
            self.vision_shapes.append(vision_shape)
        self.mouse_circle = pyglet.shapes.Circle(300, 300, 5, color=(200, 100, 255), batch=self.batch)  # Mouse circle
        self.mouse_visible = True
        self.mouse_x = 0
        self.mouse_y = 0
        self.frame_count = 0
        pyglet.clock.schedule_interval(self.update, 1/100)

    def init_bugs(self, n=20):
        bugs = []
        for _ in range(n):
            position = np.random.uniform(-3.5, 3.5, size=2)
            new_bug = Bug(position, self)
            new_bug.parent_position = None
            bugs.append(new_bug)
        return bugs


    def on_ding_end(self):
        self.current_ding_sounds -= 1  # Decrease the count when a ding sound finishes

    def synthesize_ding_sound(self):
        if self.current_ding_sounds < self.ding_sound_limit:

            duration = 0.08  # Duration of the sound in seconds
            frequency = 10000  # Starting frequency for the "ding" sound

            # Define ADSR envelope
            envelope = synthesis.ADSREnvelope(attack=1, decay=1, release=2, sustain_amplitude=1)

            # Create a sine wave with the envelope
            sine_wave = synthesis.Sine(duration=duration, frequency=frequency, envelope=envelope)

            # Play the sound
            player = Player()
            player.queue(sine_wave)
            player.play()

    def synthesize_death_sound(self):
        player = Player()
        duration = 1
        frequency = 4000
        envelope = LinearDecayEnvelope(peak=0.1)
        sawtooth_wave = StaticSource(Sawtooth(duration, frequency, envelope=envelope))
        player.queue(sawtooth_wave)
        player.play()

    def synthesize_wall_death_sound(self):
        player = Player()
        duration = 1
        frequency = 2000
        envelope = LinearDecayEnvelope(peak=0.1)
        sawtooth_wave = StaticSource(Sawtooth(duration, frequency, envelope=envelope))
        player.queue(sawtooth_wave)
        player.play()

    def synthesize_mouse_death_sound(self):
        player = Player()
        duration = 1
        frequency = 1000
        envelope = LinearDecayEnvelope(peak=0.3)
        sawtooth_wave = StaticSource(Sawtooth(duration, frequency, envelope=envelope))
        player.queue(sawtooth_wave)
        player.play()

    def update(self, dt):
        alive_bugs, new_shapes, new_vision_shapes = [], [], []
        new_parent_lines = []

        for bug, shape, vision_shape in zip(self.bugs, self.shapes, self.vision_shapes):
            if bug.alive:
                current_mouse_pos = self.get_current_mouse_pos()
                bug.move(self.bugs, current_mouse_pos, dt)

                shape.position = bug.position[0] * 75 + 300, bug.position[1] * 75 + 300
                shape.radius = bug.size * 10
                shape.color = (255, 0, 0) if bug.is_baby else (0, 0, 255)

                vision_shape.position = shape.position
                vision_shape.radius = bug.vision_range * 10

                alive_bugs.append(bug)
                new_shapes.append(shape)
                new_vision_shapes.append(vision_shape)
            else:
                shape.delete()
                vision_shape.delete()

        for line_data, fade_timer in self.parent_lines:
            # Check if fade_timer is an array or tensor and handle accordingly
            if isinstance(fade_timer, np.ndarray) or isinstance(fade_timer, torch.Tensor):
                if fade_timer.size == 1:
                    fade_timer = fade_timer.item()  # Convert to scalar if single-element
                else:
                    continue  # Avoid the ValueError

            # Check fade_timer as a scalar
            if fade_timer > 0:
                new_parent_lines.append((line_data, fade_timer - 1))

        self.parent_lines = new_parent_lines
        self.bugs, self.shapes, self.vision_shapes = alive_bugs, new_shapes, new_vision_shapes

    def on_draw(self):
        self.clear()
        self.batch.draw()  # Draw all regular shapes with the batch

        # Draw parent lines
        for (parent_line, fade_timer) in self.parent_lines:
            try:
                # Extract start_pos and end_pos from parent_line
                start_pos, end_pos = parent_line

                # If start_pos or end_pos is a scalar (not in a pair), skip drawing this line
                if not isinstance(start_pos, (list, tuple, np.ndarray)) or not isinstance(end_pos, (list, tuple, np.ndarray)):
                    continue

                # Convert numpy arrays to lists for consistency
                if isinstance(start_pos, np.ndarray):
                    start_pos = start_pos.tolist()
                if isinstance(end_pos, np.ndarray):
                    end_pos = end_pos.tolist()

                # Draw the line
                x1, y1 = start_pos[0] * 75 + 300, start_pos[1] * 75 + 300
                x2, y2 = end_pos[0] * 75 + 300, end_pos[1] * 75 + 300
                line = pyglet.shapes.Line(x1, y1, x2, y2, width=2, color=(0, 255, 0))
                opacity = int(fade_timer / 30 * 255)
                line.opacity = opacity
                line.draw()

            except (TypeError, ValueError, AssertionError):
                print("Error drawing line due to incorrect formatting of start or end position.")

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = y
        # Update mouse circle position to follow the cursor
        mouse_pos = self.get_current_mouse_pos()
        self.mouse_circle.position = mouse_pos[0] * 75 + 300, mouse_pos[1] * 75 + 300

    def get_current_mouse_pos(self):
        return np.array([self.mouse_x, self.mouse_y]) / 75 - 4

    def add_bug(self, new_bug):
        # Adds a new bug to the simulation
        self.bugs.append(new_bug)
        new_shape = pyglet.shapes.Circle(
            new_bug.position[0] * 75 + 300, new_bug.position[1] * 75 + 300, 
            10, color=(255, 0, 0) if new_bug.is_baby else (0, 0, 255), 
            batch=self.batch)
        self.shapes.append(new_shape)

        new_vision_shape = pyglet.shapes.Circle(
            new_bug.position[0] * 75 + 300, new_bug.position[1] * 75 + 300, 
            new_bug.vision_range * 10, color=(0, 255, 0, 5),
            batch=self.batch)
        self.vision_shapes.append(new_vision_shape)

if __name__ == '__main__':
    window = SimulationWindow(800, 800)
    pyglet.app.run()
