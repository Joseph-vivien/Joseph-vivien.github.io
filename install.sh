#!/bin/bash

# 币安智能杠杆交易系统安装脚本
# 作者: 高级Python工程师
# 日期: 2024-05-21

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统
check_system() {
    print_info "检查系统..."
    
    # 检查操作系统
    if [[ "$(uname)" != "Linux" ]]; then
        print_error "此脚本仅支持Linux系统"
        exit 1
    fi
    
    # 检查是否为Ubuntu
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "$ID" != "ubuntu" ]]; then
            print_warning "此脚本针对Ubuntu系统优化，其他Linux发行版可能需要手动调整"
        else
            print_info "检测到Ubuntu系统: $VERSION_ID"
        fi
    fi
    
    # 检查Python版本
    if command -v python3 &>/dev/null; then
        python_version=$(python3 --version | cut -d' ' -f2)
        print_info "检测到Python版本: $python_version"
        
        # 检查Python版本是否>=3.10
        if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 || ($(echo "$python_version" | cut -d. -f1) -eq 3 && $(echo "$python_version" | cut -d. -f2) -lt 10) ]]; then
            print_warning "推荐使用Python 3.10或更高版本"
        fi
    else
        print_error "未检测到Python 3，请先安装Python 3.10或更高版本"
        exit 1
    fi
    
    print_success "系统检查完成"
}

# 安装依赖
install_dependencies() {
    print_info "安装系统依赖..."
    
    # 更新包列表
    sudo apt update
    
    # 安装基本依赖
    sudo apt install -y build-essential libssl-dev libffi-dev python3-dev python3-pip python3-venv git
    
    # 安装TA-Lib依赖
    sudo apt install -y libta-lib0 libta-lib-dev
    
    print_success "系统依赖安装完成"
}

# 创建虚拟环境
create_venv() {
    print_info "创建Python虚拟环境..."
    
    # 创建虚拟环境
    python3 -m venv venv
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    print_success "Python虚拟环境创建完成"
}

# 安装Python依赖
install_python_packages() {
    print_info "安装Python依赖包..."
    
    # 安装TA-Lib
    pip install ta-lib
    
    # 安装其他依赖
    pip install -r requirements.txt
    
    print_success "Python依赖包安装完成"
}

# 配置系统
configure_system() {
    print_info "配置系统..."
    
    # 创建必要的目录
    mkdir -p user_data/data/historical/binance
    mkdir -p user_data/backtest_results/reports
    mkdir -p user_data/backtest_results/plots
    mkdir -p user_data/backtest_results/trades
    mkdir -p user_data/logs
    
    # 检查配置文件
    if [[ ! -f user_data/config/config.json ]]; then
        print_warning "未找到配置文件，将创建示例配置文件"
        cp user_data/config/config_example.json user_data/config/config.json
    fi
    
    # 设置权限
    chmod +x run.py
    chmod +x backtest.py
    chmod +x download_binance_data.py
    
    print_success "系统配置完成"
}

# 下载历史数据
download_historical_data() {
    print_info "是否下载历史数据? [y/N]"
    read -r download_data
    
    if [[ "$download_data" =~ ^[Yy]$ ]]; then
        print_info "下载历史数据..."
        
        # 激活虚拟环境
        source venv/bin/activate
        
        # 下载BTC/USDT数据
        python download_binance_data.py --symbols BTC/USDT --timeframes 1m 5m 15m 1h 4h --start-date 2023-01-01
        
        print_success "历史数据下载完成"
    else
        print_info "跳过历史数据下载"
    fi
}

# 设置API密钥
setup_api_keys() {
    print_info "是否设置币安API密钥? [y/N]"
    read -r setup_keys
    
    if [[ "$setup_keys" =~ ^[Yy]$ ]]; then
        print_info "请输入币安API密钥:"
        read -r api_key
        
        print_info "请输入币安API密钥:"
        read -r api_secret
        
        # 更新配置文件
        if [[ -f user_data/config/config.json ]]; then
            # 使用临时文件避免直接修改原文件
            jq --arg key "$api_key" --arg secret "$api_secret" '.exchange.key = $key | .exchange.secret = $secret' user_data/config/config.json > user_data/config/config.json.tmp
            mv user_data/config/config.json.tmp user_data/config/config.json
            print_success "API密钥已更新"
        else
            print_error "未找到配置文件，无法更新API密钥"
        fi
    else
        print_info "跳过API密钥设置"
    fi
}

# 运行测试
run_tests() {
    print_info "是否运行测试? [y/N]"
    read -r run_test
    
    if [[ "$run_test" =~ ^[Yy]$ ]]; then
        print_info "运行测试..."
        
        # 激活虚拟环境
        source venv/bin/activate
        
        # 运行单元测试
        python -m unittest discover -s tests/unit
        
        print_success "测试完成"
    else
        print_info "跳过测试"
    fi
}

# 创建启动脚本
create_startup_script() {
    print_info "创建启动脚本..."
    
    cat > start.sh << 'EOF'
#!/bin/bash
# 币安智能杠杆交易系统启动脚本

# 激活虚拟环境
source venv/bin/activate

# 启动交易系统
python run.py --mode dry_run --log-level INFO

# 如果要启动实盘交易，取消下面的注释
# python run.py --mode live --log-level INFO
EOF
    
    chmod +x start.sh
    
    print_success "启动脚本已创建: start.sh"
}

# 显示完成信息
show_completion() {
    print_success "币安智能杠杆交易系统安装完成!"
    print_info "使用以下命令启动系统:"
    echo "  ./start.sh"
    print_info "使用以下命令运行回测:"
    echo "  source venv/bin/activate"
    echo "  python backtest.py --start-date 2023-01-01 --end-date 2023-12-31"
    print_info "使用以下命令下载更多历史数据:"
    echo "  source venv/bin/activate"
    echo "  python download_binance_data.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h --start-date 2023-01-01"
    print_warning "请确保在启动实盘交易前，已正确设置API密钥并充分测试策略"
}

# 主函数
main() {
    print_info "开始安装币安智能杠杆交易系统..."
    
    check_system
    install_dependencies
    create_venv
    install_python_packages
    configure_system
    download_historical_data
    setup_api_keys
    run_tests
    create_startup_script
    show_completion
}

# 执行主函数
main
