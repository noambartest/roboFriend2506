// === הגדרות SIMON SAYS ===
#define GREEN_LED 3
#define GREEN_BUTTON 2
#define RED_LED 5
#define RED_BUTTON 6
#define YELLOW_LED 9
#define YELLOW_BUTTON 10
#define BLUE_LED 12
#define BLUE_BUTTON 13

#define NUM_COLORS 4
int leds[NUM_COLORS] = {GREEN_LED, RED_LED, YELLOW_LED, BLUE_LED};
int buttons[NUM_COLORS] = {GREEN_BUTTON, RED_BUTTON, YELLOW_BUTTON, BLUE_BUTTON};

// === הגדרות SNAKE ===
const int joyX = A0;
const int joyY = A1;
const int joySW = 8;

// === מצבים ===
enum GameMode { MODE_MENU, MODE_SNAKE, MODE_SIMON };
GameMode currentMode = MODE_MENU;

void setup() {
  Serial.begin(9600);

  // הגדרות SIMON
  for (int i = 0; i < NUM_COLORS; i++) {
    pinMode(leds[i], OUTPUT);
    pinMode(buttons[i], INPUT_PULLUP);
  }

  // הגדרות SNAKE
  pinMode(joySW, INPUT_PULLUP);
}

void loop() {
  check_mode_change();  // מחפש פקודת מצב חדשה

  switch (currentMode) {
    case MODE_SNAKE:
      snake_loop();
      break;
    case MODE_SIMON:
      simon_loop();
      break;
    case MODE_MENU:
      // לא עושה כלום
      break;
  }

  delay(10);  
}

void check_mode_change() {
  if (currentMode == MODE_MENU && Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print(">>> ");
    Serial.println(command); 

    if (command == "MODE SNAKE") {
      currentMode = MODE_SNAKE;
    } else if (command == "MODE SIMON") {
      currentMode = MODE_SIMON;
    } else if (command == "MODE MENU") {
      currentMode = MODE_MENU;
    }
  }
}

// === SIMON LOGIC ===
void simon_loop() {
  handle_serial_simon();
  handle_buttons_simon();
}

void handle_serial_simon() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.startsWith("PLAY")) {
      int idx = command.substring(5).toInt();
      if (idx >= 0 && idx < NUM_COLORS) {
        digitalWrite(leds[idx], HIGH);
        delay(400);
        digitalWrite(leds[idx], LOW);
      }
    }
  }
}

void handle_buttons_simon() {
  for (int i = 0; i < NUM_COLORS; i++) {
    if (digitalRead(buttons[i]) == LOW) {
      flashLED(i);
      while (digitalRead(buttons[i]) == LOW); 
      delay(50);
      Serial.print("BTN ");
      Serial.println(i);
    }
  }
}

void flashLED(int index) {
  digitalWrite(leds[index], HIGH);
  delay(200);
  digitalWrite(leds[index], LOW);
}

// === SNAKE LOGIC ===
void snake_loop() {
  int x = analogRead(joyX);
  int y = analogRead(joyY);
  bool pressed = digitalRead(joySW) == LOW;

  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.println(pressed ? "1" : "0");

  delay(100);
}
