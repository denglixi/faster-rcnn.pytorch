food_id = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '62', '63', '64', '65', '67', '68', '69', '70', '71', '80', '81', '82', '83', '84', '85', '86', '87', '89', '90', '91', '96', '97', '98', '99', '111', '155', '199', '211', '244', '299', '300', '311', '333', '355', '477', '522', '523', '544', '699', '700', '811', '833', '844', '900', '966',
      '977', '988', '999', '1000', '1001', '1002', '1003', '1004', '1005', '1006', '2004', '2006', '2007', '2013', '2014', '2015', '2025', '2031', '2032', '2033', '2034', '2035', '2036', '2041', '2044', '2045', '2046', '2051', '2054', '2055', '2061', '2062', '2121', '2122', '2322', '2366', '2411', '2415', '2424', '2461', '2541', '2621', '2622', '2623', '3003', '3004', '3013', '3017', '3018', '3019', '3020', '3021', '3026', '3030', '3031', '3032', '3035', '3036', '3037', '3046', '3048', '3050', '3052', '3057', '3058', '3060', '3063', '4004', '4005', '4006', '4011', '4041', '4043', '4044', '4045', '4046', '4051', '4055', '4088', '4221', '6222', ]

chnname = ['玉米', '白饭', '咸鸭蛋', '翻炒蛋', '番茄蛋', '糙米', '瘦肉', '凉拌豆腐 (小方块）', '豆腐丝/豆皮', '凉拌黑木耳', '凉拌黄瓜', '小白菜（绿）', '包菜', '苦瓜', '芹菜', '番薯葉', '凉拌豆干＆胡萝卜片', '凉拌黄瓜片＆豆腐 (素肉)', '凉皮', '凉拌猪耳', '炒面', '豆芽', '白萝卜', '长豆/豇豆', '西兰花和花菜', '酸辣羊角豆/秋葵', '花菜&黑木耳', '炒大白菜', '烩白豆腐虾仁', '鲍菇', '炒西兰花＆玉米笋＆黑木耳＆藕片/西蓝花&鱿鱼', '炒豆腐', '猪肝', '土豆炒番茄酱豆', '茄子', '滷豆腐', '鸭肉', '小炒肉（猪肉片＆青椒＆豆腐）', '凉拌猪肉片', '土豆烧鸡（大盘鸡）', '西兰花& 虾', '咖喱鸡', '红酱。酸甜肉', '肉碎', '卤鸡/红烧鸡块', '猪脚', '炖排骨', '红烧五花肉/', '黑酱猪肉', '灰褐色肉球', '清蒸鱼块', '炸鱼。整只', '炸鱼片', '章鱼，鱿鱼。O型', '红烧鱼肉', '炸鱼片，粉红酱汁', '煎蛋', '白色鱼片/酸菜鱼', '红烧猪扒（黑色肉扒/各种扒', '鸡腿', '', '麻油鸡', '炸脆鸡扒', '看reference（红烧肉饼）', '毛瓜丝', '咸菜', '冬粉', '胡辣汤', '紫菜蛋花汤', '馒头', '烧饼', '画卷', '红米粥', '牛肉土豆', '茄子炒肉 （瘦肉）', '肉+土豆+长豆？？？', '豆干', '海带', '红薯/地瓜', '土豆丝', '炒猪肉（看reference）', '螃蟹', '凉拌西红柿', 'YIH鱿鱼', '南瓜', '空心菜', '牛肚，红萝卜', 'YIH 豆芽 黃瓜斯', '红烧菜花/花菜', '蘑菇 (', '炒藕片&木耳', '肉末炒豆腐（小方块，深色）', '蛋饼（三角状，一整块）', '白菜&日本豆腐（圆形）', '扣肉（大块长条）',
           'YIH 炸鱼', 'YIH 蒸炸魚', '炸鱼块（有点像猪肉块）', '粉丝(白)＆虾仁', '土豆胡萝卜排骨汤', '卷饼（里面有菜）', '肉夹馍', '花生', '胡萝卜炒肉', '红色（辣）千张', '白色柱状的，质地类似红薯（看reference）', '烧豆腐炒肉？？？', '芋头', 'j鸡腿？ YIH11oct', '莴笋+虾', '西红柿蛋花汤', '大块猪肉？？', '香肠圆葱土豆', '不知道啥，像汤', '排骨汤', '炒豆子炒蛋', '水煮蛋', '蒸水蛋', '混合蔬菜黑木耳，豌豆，芹菜，蘑菇', 'petai豆与虾', '虾芹菜', '羊角豆', '土豆切片', '炸薯条', '圆豆腐', '麻婆豆腐', '方块豆腐', '黑酱taupok用碎肉在上面', '咖喱鸡块', 'har cheong gai', '炸鸡（半弹簧）或其他部分，黄色', '棕色酱汁鼓槌', '谷物鱼', '蒸鱼片', '油煎的鱼用桃红色和白汁', '鸡爪', '五花肉整片', '咖喱蔬菜', '非常白的蔬菜', '马铃薯条', 'taupok', '釉面浅棕色肉排', '咖喱鸡腿', '炸猪肉酱', '切碎的鸡肉淡黄酱', '煎鱼片', '猪肚整块差异酱', '炸猪肚片，橙皮', '猪肚片未炸', '猪肉片', '午餐肉', '糖醋鱼', '炸鱼', '海藻鸡（紫菜鸡）', '炸土豆饼', '炸鸡块', '五香肉卷', '鸡米花', '豆腐皮（腐竹）', '咖喱鸡', '鸡排', '红烧豆腐', '煎鸡蛋', '清蒸鱼', 'unsure, blank', '小炒猪肉', '大鸡块', '炸土豆', '鱼饼', '黑木耳肉', '木耳鸡', '炸鱼片2（arts', '烧肉', '炒饭。褐色。带冷冻菜青豆', '清汤', '酸菜。黄瓜', '烧鸡', '叉烧', '烧肉', '鸭翅膀', '烤鸭', '白鸡', '鸡爪', '辣椒酱', '长片sasame红肉', '烧焦的五花肉片，黑皮肤', ]

engname = ['corn', 'rice', 'salted egg', 'scrambled egg', 'tomato egg', 'brown rice', 'lean pork', 'tofu cubes (small)', 'beancurd skin strips (cold dish)', 'black fungus', 'cucumber',
        'xiao bai cai', 'cabbage', 'bittergourd with little bit of scrambled egg', 'celery stir fried', 'green potato leaves stir fried',
        'taukwa(tofu) strips, carrot, cucumber', '素肉 fake meat with cucumber', '凉皮 (looks like horfun/kwaytiao)', 'pig ear YIH', 'fried noodles YIH', 'beansprouts', 'white radish', 'long beans', 'caulliflower and brocolli stir fried', 'chilli okra', 'caulliflower with black fungus',
        'Fried cabbage', 'tofu and shelled prawns', 'mushroom', 'lotus root, black fungus, brocolli', 'tofu stir fried', 'pork liver', 'potato and beans', 'brinjal', 'braised tofu', 'braised duck', 'pork, green capsicum, tofu', 'cold pork', 'potato chicken', 'brocolli and shelled prawn', 'curry chicken', 'sweet and sour pork', 'minced pork', 'braised chicken', 'braised pork trotter', 'stewed pork ribs', 'braised pork belly',
        'black sauce pork', 'brown meat ball', 'YIH steamed segmented fish', 'fried whole fish', 'fried fish fillet', 'sotong', 'red sauce fish', 'fried fish fillet with pink sauce', 'fried egg', 'fish skinless small pieces', 'pork cutlet, black sauce big piece', 'whole drumstick separate bowl', '', 'sesame chicken', 'fried crispy chicken cutlet', 'red sauce cutlet', 'hairy gourd stir fried', 'salted vegetable', 'glass noodles',
        'dark soup separate bowl', 'clear soup separate bowl', 'steamed bread', 'Biscuits', 'Picture', 'Red rice porridge', 'beef potato', 'brinjal lean pork', 'Meat + potatoes + long beans ???',
        'Dried bean', 'seaweed kelp', 'sweet potato', 'potato floss', 'Fried pork (see reference)', 'crab', 'Salad with tomatoes', 'cuttlefish', 'pumpkin', 'kangkong', 'beef stomach, carrot', 'YIH beansprouts and shredded cucumber', 'Braised Cauliflower / Cauliflower', 'dong gu mushrooms', 'Fried oysters and fungus', 'Fried pork with tofu (small cubes, dark)', 'Quiche (triangular, one piece)', 'Cabbage & Japanese Tofu (round)', 'Buckle meat (large strips)',
        'YIH fried segmented fish', 'YIH fried and then steamed segmented fish', 'Fried fish fillet (a bit like pork)', 'Fans (White) & Shrimp', 'Potato carrot ribs soup', 'Burrito (with vegetables inside)', 'Meat folder', 'peanut', 'Carrot fried meat', 'Red (spicy) thousand sheets', 'White columnar, texture similar to sweet potato (see reference)', 'Fried tofu fried meat ???', 'taro', 'j chicken legs? YIH11oct', 'Lettuce + Shrimp',
        'Tomato egg soup', 'Big chunks of pork?', 'Sausage, round onion, potato',
        'I dont know, like soup', 'Pork ribs soup', 'scrambled egg with baked beans', 'hard boiled egg', 'steamed water egg', 'mixed vegetable-black fungus, peas, celery, mushroom', 'petai beans with prawn', 'prawn celery', 'okra', 'potato slices', 'french fries', 'round tofu', 'mapo tofu', 'cube tofu', 'black sauce taupok with minced meat on top', 'curry chicken chunks', 'har cheong gai', 'fried chicken(half spring) or other parts, yellow', 'brown sauce drumstick', 'cereal fish', 'steamed fish fillet', 'fried fish with pink and white sauce', 'chicken feet', 'pork belly whole piece', 'curry vegetables', 'very white vegetable', 'potato strips', 'taupok', 'glazed light brown meat cutlet', 'curry chicken drumstick', 'fried pork brown sauce', 'chopped chicken yellowish sauce', 'fried version of fish fillet', 'pork belly whole piece diff sauce', 'fried pork belly slices, orange skin', 'pork belly slices unfried', 'pork slices', 'luncheon meat', 'sweet and sour fish', 'fried fish', 'seaweed chicken', 'hashbrown', 'nugget', 'ngioh hiang', 'chicken popcorn', 'beancurd skin', 'curry chicken', 'chicken cutlet', 'braised beancurd', 'fried egg', 'steamed fish', '17', 'Small fried pork', 'Big chicken', 'Fried potatoes', 'Fish cake', 'Black fungus meat', 'Fungus chicken', 'Fish fillet 2 (first)', 'roasted pork', 'fried rice', 'soup', 'acar', 'roasted chicken', 'char siew', 'roasted pork', 'duck wing', 'roasted duck', 'steamed chicken', 'chicken feet', 'chilli', 'long piece sasame red meat', 'charred pork belly slices, black skin', ]


id2eng= dict(zip(food_id, engname))
id2chn = dict(zip(food_id, chnname))