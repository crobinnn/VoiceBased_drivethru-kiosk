import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'package:flutter/services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]).then((_) {
    runApp(OrderUI());
  });
}

class OrderUI extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Order System',
      home: OrderHomePage(),
    );
  }
}

// Enhanced OrderItem class for better state management
class OrderItem {
  final String id;
  final String name;
  int quantity;
  final int price;
  Color? quantityBlinkColor;
  
  OrderItem({
    required this.id,
    required this.name,
    required this.quantity,
    required this.price,
    this.quantityBlinkColor,
  });
  
  OrderItem copyWith({
    String? id,
    String? name,
    int? quantity,
    int? price,
    Color? quantityBlinkColor,
  }) {
    return OrderItem(
      id: id ?? this.id,
      name: name ?? this.name,
      quantity: quantity ?? this.quantity,
      price: price ?? this.price,
      quantityBlinkColor: quantityBlinkColor ?? this.quantityBlinkColor,
    );
  }
}

class OrderHomePage extends StatefulWidget {
  @override
  _OrderHomePageState createState() => _OrderHomePageState();
}

class _OrderHomePageState extends State<OrderHomePage>
    with TickerProviderStateMixin {
  WebSocketChannel? _channel;
  StreamSubscription? _streamSubscription;
  
  // Enhanced order management with AnimatedList support
  List<OrderItem> orderItems = [];
  final GlobalKey<AnimatedListState> _listKey = GlobalKey<AnimatedListState>();
  
  String systemStatus = "Kendaraan tidak terdeteksi";
  bool isListening = false;
  bool isConfirming = false;
  int totalPrice = 0;
  
  // Connection state
  bool _showInitializingPopup = true;
  int _dotCount = 0;
  bool _controllerConnected = false;
  bool _inferenceConnected = false;
  Timer? _dotTimer;
  Timer? _connectionTimer;
  Timer? _reconnectTimer;
  bool _isConnectionActive = false;

  // Popup states
  bool _showOrderConfirmedPopup = false;
  bool _showOrderCanceledPopup = false;
  String _orderConfirmedMessage = "";
  String _orderCanceledMessage = "";

  // Valid menu items
  final List<String> _validMenus = ['ayam', 'es jeruk', 'sosis', 'es krim', 'roti'];

  // Track blink timers to prevent memory leaks
  Map<String, Timer> _blinkTimers = {};

  @override
  void initState() {
    super.initState();
    _startDotAnimation();
    _initConnection();
  }

  void _startDotAnimation() {
    _dotTimer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      if (mounted) {
        setState(() {
        _dotCount = (_dotCount + 1) % 4;
      });
      }
    });
  }

  void _initConnection() {
    _connectToWebSocket();
    // Setup periodic connection check
    _connectionTimer = Timer.periodic(Duration(seconds: 5), (_) {
      if (!_isConnectionActive && mounted) {
        _connectToWebSocket();
      }
    });
  }

  Future<void> _connectToWebSocket() async {
    if (_isConnectionActive) return;

    try {
      // Close existing connection if any
      await _disconnect();

      if (mounted) {
        setState(() {
        _controllerConnected = false;
        _inferenceConnected = false;
      });
      }

      _channel = WebSocketChannel.connect(
        Uri.parse('ws://0.0.0.0:33941'),
      );

      _streamSubscription = _channel!.stream.listen(
        _handleMessage,
        onError: (error) => _handleDisconnect(error),
        onDone: () => _handleDisconnect('Connection closed'),
      );

      if (mounted) setState(() {
        _isConnectionActive = true;
      });

    } catch (e) {
      _handleDisconnect(e.toString());
    }
  }

  void _handleMessage(dynamic message) {
    if (!mounted) return;

    final decoded = json.decode(message);
    print('Received message: $decoded');

    setState(() {
      // Handle multiple orders first
      if (decoded['orders'] is List) {
        for (var order in decoded['orders']) {
          _processSingleOrder(order);
        }
        return;
      }
      
      // Fallback to single order processing
      _processSingleOrder(decoded);
    });
  }

  void _processSingleOrder(Map<String, dynamic> order) {
    switch (order['type'] ?? order['intent']) {
      case 'ready':
        _controllerConnected = true;
        print("Server is ready!");
        break;

      case 'inference_ok':
        _inferenceConnected = true;
        systemStatus = "Kendaraan tidak terdeteksi";
        Future.delayed(Duration(milliseconds: 500), () {
          if (mounted) setState(() {
            _showInitializingPopup = false;
          });
        });
        break;

      case 'car_detected':
        systemStatus = order['message'] ?? "Kalibrasi Mic...";
        break;
      
      case 'invalid_menu':
        systemStatus = order['message'] ?? "Beberapa menu yang anda pesan tidak tersedia";
        break;

      case 'start_speak':
        isListening = false;
        isConfirming = false;
        systemStatus = order['message'] ?? "Mulai bicara untuk memesan";
        break;

      case 'start_listening':
        isListening = true;
        isConfirming = false;
        systemStatus = "Mendengarkan pesanan anda...";
        break;

      case 'processing_order':
        isListening = false;
        isConfirming = false;
        systemStatus = order['message'] ?? "Memproses pesanan anda...";
        break;

      case 'add_item':
        _handleAddOrder(order['items']);
        systemStatus = order['message'] ?? "Pesanan diperbarui, mohon cek kembali pesanan anda";
        break;

      case 'remove_item':
        _handleRemoveOrder(order['items']);
        systemStatus = order['message'] ?? "Pesanan diperbarui, mohon cek kembali pesanan anda";
        break;

      case 'incomplete_change':
        systemStatus = order['message'] ?? "Penggantian pesanan tidak dapat diproses, untuk mengganti pesanan mohon sebutkan menu yang ingin diganti dan menu penggantinya";
        break;

      case 'item_not_in_order':
        systemStatus = order['message'] ?? "Pesanan tidak dapat diproses, karena ada menu yang belum anda tambahkan pada pesanan";
        break;
      
      case 'insufficient_quantity':
        systemStatus = order['message'] ?? "Penggantian pesanan tidak dapat diproses, jumlah menu yang diganti melebihi jumlah yang ada pada pesanan saat ini";
        break;
        
      case 'order_confirmed':
        _showOrderConfirmedPopup = true;
        _orderConfirmedMessage = order['message'] ?? "Pesanan selesai";
        systemStatus = "Pesanan selesai";
        isListening = false;
        isConfirming = false;
        break;

      case 'order_canceled':
        _showOrderCanceledPopup = true;
        _orderCanceledMessage = order['message'] ?? "Pesanan dibatalkan";
        systemStatus = "Pesanan dibatalkan";
        isListening = false;
        isConfirming = false;
        break;

      case 'no_intent':
        systemStatus = order['message'] ?? "Maaf intensi anda kurang jelas mohon berbicara kembali";
        isListening = false;
        isConfirming = false;
        break;
        
      case 'car_gone':
        _handleCarGone();
        break;

      case 'error':
        systemStatus = "Error: ${order['message']}";
        
      case 'inference_disconnect':
        systemStatus = "Error: ${order['message']}";
        _showInitializingPopup = true;
        _inferenceConnected = false;
        isListening = false;
        isConfirming = false;
        break;
    }
  }
  
  void _handleCarGone() {
    _showOrderConfirmedPopup = false;
    _showOrderCanceledPopup = false;
    
    // Cancel all blink timers
    _blinkTimers.values.forEach((timer) => timer.cancel());
    _blinkTimers.clear();
    
    // Animate all items out before clearing
    for (int i = orderItems.length - 1; i >= 0; i--) {
      _removeItemAnimated(i);
    }
    
    totalPrice = 0;
    systemStatus = "Kendaraan tidak terdeteksi";
    isListening = false;
    isConfirming = false;
  }

  void _handleAddOrder(Map<String, dynamic> items) {
    items.forEach((menu, qty) {
      if (_validMenus.contains(menu)) {
        int quantity = qty;
        int existingIndex = orderItems.indexWhere((item) => item.name == menu);
        
        if (existingIndex != -1) {
          // Update existing item with blink animation
          _updateItemQuantity(existingIndex, orderItems[existingIndex].quantity + quantity);
          _blinkQuantityText(existingIndex, Colors.green);
        } else {
          // Add new item with smooth animation
          _addItemAnimated(menu, quantity);
        }
        
        totalPrice += getPrice(menu) * quantity;
      }
    });
  }

  void _handleRemoveOrder(Map<String, dynamic> items) {
    items.forEach((menu, qty) {
      if (_validMenus.contains(menu)) {
        int quantity = qty;
        int existingIndex = orderItems.indexWhere((item) => item.name == menu);
        
        if (existingIndex != -1) {
          int newQuantity = (orderItems[existingIndex].quantity - quantity).clamp(0, double.infinity).toInt();
          totalPrice -= getPrice(menu) * quantity;
          
          if (newQuantity == 0) {
            // Remove item with smooth animation
            _removeItemAnimated(existingIndex);
          } else {
            // Update quantity with blink animation
            _updateItemQuantity(existingIndex, newQuantity);
            _blinkQuantityText(existingIndex, Colors.red);
          }
        }
      }
    });
  }

  void _addItemAnimated(String menu, int quantity) {
    final newItem = OrderItem(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: menu,
      quantity: quantity,
      price: getPrice(menu),
    );
    
    orderItems.add(newItem);
    _listKey.currentState?.insertItem(
      orderItems.length - 1,
      duration: Duration(milliseconds: 500),
    );
  }

  void _removeItemAnimated(int index) {
    if (index < 0 || index >= orderItems.length) return;
    
    final removedItem = orderItems[index];
    
    // Cancel any existing blink timer for this item
    _blinkTimers[removedItem.id]?.cancel();
    _blinkTimers.remove(removedItem.id);
    
    orderItems.removeAt(index);
    
    _listKey.currentState?.removeItem(
      index,
      (context, animation) => _buildAnimatedTile(removedItem, animation, true),
      duration: Duration(milliseconds: 300),
    );
  }

  void _updateItemQuantity(int index, int newQuantity) {
    if (index < 0 || index >= orderItems.length) return;
    
    setState(() {
      orderItems[index] = orderItems[index].copyWith(quantity: newQuantity);
    });
  }

  void _blinkQuantityText(int index, Color blinkColor) {
    if (index < 0 || index >= orderItems.length) return;
    
    final item = orderItems[index];
    
    // Cancel any existing blink timer for this item
    _blinkTimers[item.id]?.cancel();
    
    // Set the blink color
    setState(() {
      orderItems[index] = orderItems[index].copyWith(
        quantityBlinkColor: blinkColor,
      );
    });
    
    // Set timer to clear blink color after 1 second
    _blinkTimers[item.id] = Timer(Duration(seconds: 1), () {
      if (mounted) {
        setState(() {
          // Find the item by ID and clear its blink color
          final currentIndex = orderItems.indexWhere((i) => i.id == item.id);
          if (currentIndex != -1) {
            orderItems[currentIndex] = orderItems[currentIndex].copyWith(quantityBlinkColor: null);
          }
        });
        _blinkTimers.remove(item.id);
      }
    });
  }
  
  void _handleDisconnect(String reason) {
    print('Disconnected: $reason');
    if (!mounted) return;

    setState(() {
      _isConnectionActive = false;
      _controllerConnected = false;
      _inferenceConnected = false;
      isListening = false;
      systemStatus = "Connection lost - Reconnecting...";
      _showInitializingPopup = true;
    });

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(Duration(seconds: 2), _connectToWebSocket);
  }

  Future<void> _disconnect() async {
    _reconnectTimer?.cancel();
    await _streamSubscription?.cancel();
    await _channel?.sink.close(ws_status.goingAway);
  }

  int getPrice(String menu) {
    final prices = {
      'ayam': 25000,
      'es jeruk': 30000,
      'sosis': 5000,
      'es krim': 15000,
      'roti': 20000,
    };
    return prices[menu] ?? 0;
  }

  Color _getStatusColor() {
    if (isListening) return Colors.blue[100]!;
    if (systemStatus.contains("selesai")) return Colors.green[100]!;
    if (systemStatus.contains("dibatalkan")) return Colors.red[100]!;
    if (systemStatus.contains("Memproses")) return Colors.yellow[100]!;
    return Colors.grey[200]!;
  }

  Color _getStatusTextColor() {
    if (isListening) return Colors.blue[800]!;
    if (systemStatus.contains("selesai")) return Colors.green[800]!;
    if (systemStatus.contains("dibatalkan")) return Colors.red[800]!;
    if (systemStatus.contains("Memproses")) return Colors.yellow[800]!;
    return Colors.grey[800]!;
  }

  Widget _buildPopupDialog(BuildContext context, {
    required String title,
    required String message,
    required IconData icon,
    required Color iconColor,
  }) {
    return Container(
      color: Colors.black.withValues(alpha:0.7),
      child: Center(
        child: Container(
          width: MediaQuery.of(context).size.width * 0.8,
          padding: EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(15)),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 60, color: iconColor),
              SizedBox(height: 20),
              Text(
                title,
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 10),
              Text(
                message,
                style: TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 20),
              Text("Terima Kasih!",
                style: TextStyle(
                  fontSize: 16,
                  fontStyle: FontStyle.italic,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusRow(String text, bool connected) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          connected ? text : "$text...",
          style: TextStyle(fontSize: 16),
        ),
        SizedBox(width: 8),
        if (connected)
          Icon(Icons.check_circle, color: Colors.green, size: 20)
        else
          SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
      ],
    );
  }

  // Enhanced animated tile builder with smooth fade-in/fade-out only
  Widget _buildAnimatedTile(OrderItem item, Animation<double> animation, [bool isRemoving = false]) {
    final scaleAnimation = Tween<double>(
      begin: isRemoving ? 1.0 : 0.8,
      end: isRemoving ? 0.8 : 1.0,
    ).animate(CurvedAnimation(
      parent: animation,
      curve: isRemoving ? Curves.easeInBack : Curves.elasticOut,
    ));

    final fadeAnimation = Tween<double>(
      begin: isRemoving ? 1.0 : 0.0,
      end: isRemoving ? 0.0 : 1.0,
    ).animate(CurvedAnimation(
      parent: animation,
      curve: Curves.easeInOut,
    ));

    return ScaleTransition(
      scale: scaleAnimation,
      child: FadeTransition(
        opacity: fadeAnimation,
        child: _buildTile(item),
      ),
    );
  }

  // Enhanced tile builder with quantity blink animation
  Widget _buildTile(OrderItem item) {
    final menuImages = {
      'ayam': 'assets/ayam.png',
      'es jeruk': 'assets/kopi.png',
      'sosis': 'assets/sosis.png',
      'es krim': 'assets/cream.png',
      'roti': 'assets/roti.png',
    };
    
    return Container(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha:0.1),
            blurRadius: 6,
            offset: Offset(0, 3),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha:0.05),
            blurRadius: 2,
            offset: Offset(0, 1),
          ),
        ],
      ),
      child: ListTile(
        leading: Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            image: DecorationImage(
              image: AssetImage(menuImages[item.name] ?? ''),
              fit: BoxFit.cover,
            ),
          ),
        ),
        title: Text(
          item.name.toUpperCase(),
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 16,
            color: Colors.grey[800],
          ),
        ),
        subtitle: Row(
          children: [
            Text(
              "Quantity: ",
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
            // Animated quantity counter with blink effect
            TweenAnimationBuilder<int>(
              tween: IntTween(begin: 0, end: item.quantity),
              duration: Duration(milliseconds: 600),
              curve: Curves.easeInOut,
              builder: (context, value, child) {
                return AnimatedContainer(
                  duration: Duration(milliseconds: 200),
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: item.quantityBlinkColor?.withValues(alpha: 0.3),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    value.toString(),
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: item.quantityBlinkColor ?? Colors.grey[800],
                    ),
                  ),
                );
              },
            ),
          ],
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              _formatPrice(item.price),
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
            // Animated total price
            TweenAnimationBuilder<int>(
              tween: IntTween(begin: 0, end: item.price * item.quantity),
              duration: Duration(milliseconds: 600),
              curve: Curves.easeInOut,
              builder: (context, value, child) {
                return Text(
                  _formatPrice(value),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: const Color.fromARGB(255, 0, 0, 0),
                  ),
                );
              },
            ),
          ],
        ),
        contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      ),
    );
  }

  String _formatPrice(int price) {
    return "${(price / 1000).toStringAsFixed(0)}k";
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final titleContainerHeight = 80.0;
    
    return Scaffold(
      backgroundColor: Colors.grey[100],
      body: Stack(
        children: [
          Column(
            children: [
              // Drive Thru Title Header
              Container(
                height: titleContainerHeight,
                padding: EdgeInsets.symmetric(vertical: 16),
                decoration: BoxDecoration(
                  color: const Color.fromARGB(255, 0, 0, 0),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black26,
                      blurRadius: 10,
                      offset: Offset(0, 4),
                    )
                  ],
                ),
                child: Center(
                  child: Text(
                    "DRIVE-THRU KIOSK",
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: 2.0,
                      shadows: [
                        Shadow(
                          blurRadius: 10.0,
                          color: Colors.black45,
                          offset: Offset(2.0, 2.0),)
                      ],
                    ),
                  ),
                ),
              ),
              
              // Order list section (75% of remaining space)
              SizedBox(
                height: (screenHeight - titleContainerHeight) * 0.75,
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "Order List:", 
                        style: TextStyle(
                          fontSize: 20, 
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[800],
                        ),
                      ),
                      SizedBox(height: 10),
                      Expanded(
                        // Replace ListView with AnimatedList
                        child: AnimatedList(
                          key: _listKey,
                          initialItemCount: orderItems.length,
                          itemBuilder: (context, index, animation) {
                            if (index >= orderItems.length) return SizedBox.shrink();
                            return _buildAnimatedTile(orderItems[index], animation);
                          },
                        ),
                      ),
                      Divider(),
                      Align(
                        alignment: Alignment.centerRight,
                        child: TweenAnimationBuilder<int>(
                          tween: IntTween(begin: 0, end: totalPrice),
                          duration: Duration(milliseconds: 800),
                          curve: Curves.easeInOut,
                          builder: (context, value, child) {
                            return Text(
                              "Subtotal: ${_formatPrice(value)}", 
                              style: TextStyle(
                                fontSize: 18, 
                                fontWeight: FontWeight.w600,
                                color: Colors.grey[800],
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // Status section (25% of remaining space)
              Container(
                height: (screenHeight - titleContainerHeight) * 0.25,
                padding: EdgeInsets.all(16),
                width: double.infinity,
                decoration: BoxDecoration(
                  color: _getStatusColor(),
                  borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (isListening)
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.graphic_eq, color: Colors.blue, size: 40),
                          SizedBox(height: 16),
                          Text(
                            "Mendengarkan Pesanan Anda...", 
                            style: TextStyle(
                              fontSize: 24, 
                              fontWeight: FontWeight.bold,
                              color: Colors.blue[800],
                            ),
                          ),
                        ],
                      )
                    else
                      Expanded(
                        child: Center(
                          child: Text(
                            systemStatus, 
                            style: TextStyle(
                              fontSize: 24, 
                              fontWeight: FontWeight.bold,
                              color: _getStatusTextColor(),
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ),
                  ],
                ),
              )
            ],
          ),

          if (_showInitializingPopup)
            Container(
              color: Colors.black.withValues(alpha:0.7),
              child: Center(
                child: Container(
                  width: MediaQuery.of(context).size.width * 0.8,
                  padding: EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(15)),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        "Initializing${'.' * _dotCount}",
                        style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 30),
                      _buildStatusRow(
                        "Controller Connection",
                        _controllerConnected,
                      ),
                      SizedBox(height: 10),
                      _buildStatusRow(
                        "Inference Server",
                        _inferenceConnected,
                      ),
                      if (!_isConnectionActive) ...[
                        SizedBox(height: 20),
                        Text(
                          "Connecting...",
                          style: TextStyle(color: Colors.grey),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),

          if (_showOrderConfirmedPopup && !_showInitializingPopup)
            _buildPopupDialog(
              context,
              title: "Pesanan Selesai",
              message: _orderConfirmedMessage,
              icon: Icons.check_circle,
              iconColor: Colors.green,
            ),

          if (_showOrderCanceledPopup && !_showInitializingPopup)
            _buildPopupDialog(
              context,
              title: "Pesanan Dibatalkan",
              message: _orderCanceledMessage,
              icon: Icons.cancel,
              iconColor: Colors.red,
            ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _dotTimer?.cancel();
    _connectionTimer?.cancel();
    _reconnectTimer?.cancel();
    
    // Cancel all blink timers
    _blinkTimers.values.forEach((timer) => timer.cancel());
    _blinkTimers.clear();
    
    _disconnect();
    super.dispose();
  }
}