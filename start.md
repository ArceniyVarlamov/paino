# Обычный пользовательский вход
python3 /Users/arceniy/Documents/Projects/Piano/interactive_tester.py

# Рахманов
python3 /Users/arceniy/Documents/Projects/Piano/calibrate_hybrid_profile.py \
  /Users/arceniy/Documents/Projects/Piano/midi/rach_solo.json

python3 /Users/arceniy/Documents/Projects/Piano/interactive_tester.py \
  /Users/arceniy/Documents/Projects/Piano/midi/rach_solo.json \
  --orchestra-midi /Users/arceniy/Documents/Projects/Piano/midi/rach_orchestra.mid \
  --midi-out 3 \
  --orchestra-midi-channel 2 \
  --orchestra-volume 0.85



# Руками
python3 /Users/arceniy/Documents/Projects/Piano/calibrate_hybrid_profile.py \
  /Users/arceniy/Documents/Projects/Piano/midi/left_hand.json


python3 /Users/arceniy/Documents/Projects/Piano/interactive_tester.py \
  "/Users/arceniy/Documents/Projects/Piano/midi/in_the_pool_–_Chainsaw_Man_The_Movie__Reze_Arc_OST.mid" \
  --practice-hand left \
  --midi-out 2 \
  --orchestra-midi-channel 2 \
  --piano-midi-out 3 \
  --piano-midi-channel 1 \
  --mute-local-piano


# Импорт нового MIDI в библиотеку проекта
# Если путь не передан, откроется native file picker.
python3 /Users/arceniy/Documents/Projects/Piano/midi_workspace.py

# Импорт конкретного файла
python3 /Users/arceniy/Documents/Projects/Piano/midi_workspace.py \
  /Users/arceniy/Documents/Projects/Piano/midi/rach_solo.mid

# Посмотреть уже импортированные workspace'ы
python3 /Users/arceniy/Documents/Projects/Piano/midi_workspace.py --list
